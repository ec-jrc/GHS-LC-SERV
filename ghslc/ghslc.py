from typing import Iterable, List
from pathlib import Path
import zipfile
from osgeo import gdal
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import affine
import numpy as np
from PIL import Image
from skimage.morphology import binary_erosion
from scipy.ndimage import gaussian_filter
import warnings
import yaml


# set GDAL the python way
gdal.UseExceptions()
gdal.SetConfigOption('GDAL_CACHEMAX', '512')

# set print as standard log method
logs = print


def read_s2_bands_as_vrt(filename_safe: Path, pixres: int, out_vrt: Path) -> None:
    """
    Create a VRT dataset based on S2 bands

    :param filename_safe: Path
        the complete filename of the Sentinel 2 SAFE product
    :param pixres: int
        the pixel size in meters used to filter bands:
        - 10: 'B02', 'B03', 'B04', 'B08'
        - 20: 'B05', 'B06', 'B07', 'B11', 'B12', 'B8A'

    :param out_vrt: Path
        the complete filename to use for the resulting VRT
    """

    if filename_safe.suffix == '.zip':
        with zipfile.ZipFile(filename_safe) as zf:
            jp2s_all = [f'/vsizip/{filename_safe}/{zip_file}' for zip_file in zf.namelist() if zip_file.endswith('.jp2')]
    else:
        jp2s_all = [str(file) for file in filename_safe.rglob('*.jp2')]

    if pixres == 10:
        bands = ['B02', 'B03', 'B04', 'B08']
    else:
        bands = ['B05', 'B06', 'B07', 'B11', 'B12', 'B8A']

    jp2s = [jp2 for jp2 in jp2s_all for band in bands if f'{band}.jp2' in jp2]

    gdal.BuildVRT(
        destName=str(out_vrt),
        srcDSOrSrcDSTab=jp2s,
        separate=True, srcNodata=0, VRTNodata=0,
    )


def split_domain(filename: Path, pixres: int) -> (np.ndarray, np.ndarray):
    """
    Split the data in two domains A and B based on luminance

    :param filename: Path
        the filename of the dataset composed by S2 10m bands
    :param pixres: int
        the pixel resolution to filter Sentinel 2 bands, it can be 10 or 20 (meters)

    :return: (np.ndarray, np.ndarray)
        the two domains A and B
    """

    with rasterio.open(filename) as vrt:
        data = vrt.read()

    # data domain
    domain = data.min(axis=0) > 0
    # luminance
    luminance = data[:3, :, :].max(axis=0)
    # automatic threshold (OTSU)
    thr_otsu = threshold_otsu(luminance[domain])

    domain_a = np.logical_and(domain, luminance <= thr_otsu)
    domain_b = np.logical_and(domain, luminance > thr_otsu)

    if pixres == 20:
        shape_20m = [int(ds / 2) for ds in domain.shape]
        domain_a = np.array(Image.fromarray(domain_a).resize(shape_20m, Image.NEAREST))
        domain_b = np.array(Image.fromarray(domain_b).resize(shape_20m, Image.NEAREST))

    return domain_a, domain_b


def threshold_otsu(image: np.ndarray) -> int:
    """
    Compute image threshold using Otsu’s method as done in MATLAB's multithresh

    Check the official documentation: https://it.mathworks.com/help/images/ref/multithresh.html
    The code is based on implementation for R2020b version, but it compute a single threshold value

    :param image: np.ndarray
        the image to threshold

    :return: int
        the threshold value
    """

    # Variables are named similar to the formulae in Otsu's paper.
    min_image = image.min()
    max_image = image.max()
    if min_image == max_image:
        raise ValueError('Values must be different! min = {}, max = {}'.format(min_image, max_image))

    image = np.reshape(image, (-1, 1), order='F')
    image = (np.asarray(image, dtype=np.single) - min_image) / (max_image - min_image)
    image = np.round(image / image.max() * 255).astype(np.uint8)

    counts = np.bincount(image.flat, minlength=256)
    p = counts / counts.sum()

    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(start=1, stop=257))
    mu_t = mu[-1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma_b_squared = np.power(mu_t * omega - mu, 2) / (omega * (1 - omega))

    maxval = np.nanmax(sigma_b_squared)
    idx = np.where(sigma_b_squared == maxval)
    # Find the intensity associated with the bin
    thresh = np.mean(idx)
    # Map to original scale
    scale_thresh = min_image + (thresh / 255 * (max_image - min_image))
    scale_thresh = np.round(scale_thresh).astype(min_image.dtype)

    return scale_thresh


def read_ancillary_data(filename_ancillary: Path, crs: str, bounds: Iterable[float],
                        width: int, height: int) -> np.ndarray:
    """
    Read ancillary dataset as numpy array using a target spatial extent

    :param filename_ancillary: str
        the complete filename of the ancillary dataset
    :param crs: str
        the target coordinate reference systems
    :param bounds: Iterable[float]
        the target spatial extent used to clip
    :param width: int
        the target dataset width in number of pixels
    :param height: int
        the target dataset height in number of pixels

    :return: np.ndarray
        the ancillary data warped and clip as a numpy array
    """

    # create dataset in memory
    # filename_memory = '/vsimem/ancillary.tif'
    logs(f'Ancillary file: {filename_ancillary}')
    filename_memory = filename_ancillary.parent / 'ancillary.tif'
    logs(f'Warped ancillary: {filename_memory}')
    gdal.Warp(
        str(filename_memory),
        str(filename_ancillary),
        format='GTiff',
        dstSRS=crs,
        outputBounds=bounds,
        width=width,
        height=height,
    )

    with rasterio.open(filename_memory) as src:
        data = src.read(1)

    return data


def data_quantile(datafile: Path, domain_a: np.ndarray, domain_b: np.ndarray, nlevels: int,
                  output: Path) -> (Path, Path):
    """
    Quantile data in two domains with in a given number of levels

    :param datafile: Path
        the complete filename of the dataset to quantile
    :param domain_a: np.ndarray
        the domain A
    :param domain_b: np.ndarray
        the domain B
    :param nlevels: int
        the number of levels used to quantile

    """

    logs('Quantile data with levels: {}'.format(nlevels))

    with rasterio.open(datafile) as src:
        data = src.read()
        profile = src.profile
        profile.update(
            driver='GTiff',
            compress='lzw',
            dtype=np.uint8,
            blockxsize=256,
            blockysize=256,
        )

    saturation = 0.0001
    low_q = saturation
    high_q = 1 - saturation

    dst_file_a = output / f'{datafile.stem}_domA_quant.tif'
    dst_file_b = output / f'{datafile.stem}_domB_quant.tif'
    with rasterio.open(dst_file_a, 'w', **profile) as dst_a:
        with rasterio.open(dst_file_b, 'w', **profile) as dst_b:
            for i, band in enumerate(data, start=1):
                logs('quantile band: {}'.format(i))

                q_min_a = np.quantile(band[domain_a], low_q, interpolation='lower')
                q_max_a = np.quantile(band[domain_a], high_q, interpolation='higher')
                x_norm_a = np.round(imrscl(data=band, min_value=q_min_a, max_value=q_max_a) * (nlevels - 2))

                dst_a.write(x_norm_a, i)

                q_min_b = np.quantile(band[domain_b], low_q, interpolation='lower')
                q_max_b = np.quantile(band[domain_b], high_q, interpolation='higher')
                x_norm_b = np.round(imrscl(data=band, min_value=q_min_b, max_value=q_max_b) * (nlevels - 2))

                dst_b.write(x_norm_b, i)

    return dst_file_a, dst_file_b


def imrscl(data: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """
    Rescale data values between given min and max

    :param data: np.ndarray
        the data to be rescaled
    :param min_value: float
        the minimum value
    :param max_value: float
        the maxium value

    :return: np.ndarray
        the input data rescaled between min and max input values
    """

    data = np.asarray(data, dtype=np.double)
    min_value = np.double(min_value)
    max_value = np.double(max_value)

    if min_value < max_value:
        out = np.minimum(np.maximum(data, min_value), max_value)
        out = out - min_value
        out = out / (max_value - min_value)
    else:
        out = (data >= max_value).astype(np.double)

    out[np.isnan(data)] = np.nan

    return out


def sml_minimal_support_multiple_quantization(datafile: Path, levels: int, minimal_support: int,
                                              multiple_quantization: List[int]) -> (np.ndarray, List[np.ndarray]):
    """
    Data sequence encoding minimal-support multiple-quantization

    :param datafile: Path
        the complete filename of the dataset to encode
    :param levels: int
        the number of levels used to encode
    :param minimal_support: int
        the minimal support value
    :param multiple_quantization: List[int]
        the quantization values

    :return: (np.ndarray, List[np.ndarray])
        the solved domain and the quantized datastack
    """

    with rasterio.open(datafile) as src:
        datastack = src.read()

    _, rows, cols = datastack.shape
    solved_domain = np.zeros(shape=(rows, cols), dtype=np.uint8)

    datastack_quantized = []

    for i, quant in enumerate(multiple_quantization, start=1):
        logs('encode sequence {} with quantization: {}'.format(i, quant))
        layer_quantized = np.floor(np.divide(datastack, quant, dtype=np.double))
        data_sequenced = sml_sequence_encode(layer_quantized, levels)

        data_sequenced = np.unique(data_sequenced, return_inverse=True)[1]
        # matlab style, add 1 to start from 1 instead of 0
        data_sequenced = np.reshape(data_sequenced + 1, newshape=(rows, cols))
        datastack_quantized.append(data_sequenced)

        data_support = sml_histogram_count(data=data_sequenced, num_bins=data_sequenced.max())

        # create the hierarchical minimal support decision map
        data_unique_index_support = data_support[data_sequenced - 1]
        new_solution_domain = np.logical_and(data_unique_index_support >= minimal_support, solved_domain == 0)

        # again, matlab style index value starting from 1
        solved_domain[new_solution_domain] = i

    # force to convergence at the greatest Q of residual sequences with less support at any Q
    solved_domain[solved_domain == 0] = len(multiple_quantization)

    return solved_domain, datastack_quantized


def sml_sequence_encode(x: np.ndarray, base: int) -> np.ndarray:
    """
    Encodes the sequences used for symbolic machine learning (SML) classification

    :param x: np.ndarray
        the input data to be encoded, it can be:
        - 2D input matrix: 2D table, the 2nd dimension is the layer/band
        - 3D input matrix: 3D stack of images, the 3rd dimension is the layer/band
    :param base: int
        the numeric base used for encoding. Note must be base > nlevel
        with nlevel equal to the number of levels in the input data

    :return: np.ndarray
        the encoded sequences
    """

    num_levels = x.max() + 1
    if num_levels > base:
        raise ValueError('Number of data levels {} must be less than base {}'.format(num_levels, base))

    # 3D input matrix: 3D stack of images, it could be a single layer/band (2D) image
    if len(x.shape) > 2:
        num_layers, height, width = x.shape
    else:
        num_layers = 1
        height, width = x.shape

    num_sample = width * height
    x = np.reshape(x, newshape=(num_layers, num_sample))

    # convert the data to the specified base
    exponents = np.arange(num_layers, dtype=np.uint64)
    powers = np.power(base, exponents, dtype=np.uint64)

    sequence_encoded = np.zeros(shape=num_sample, dtype=np.uint64)
    for ii in range(num_layers):
        sequence_encoded += np.multiply(x[ii, :], powers[ii], dtype=np.uint64, casting='unsafe')

    if sequence_encoded.max() > np.iinfo(np.uint64).max:
        raise ValueError('Not enough numbers for encoding all the potential sequences!'
                         '\nReduce the number of levels or the number of layers')

    sequence_encoded = np.reshape(sequence_encoded, newshape=(height, width))

    return sequence_encoded


def sml_histogram_count(data: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Count number of occurrences of each value for symbolic machine learning (SML) classification

    :param data: np.ndarray of int
        The data to be counted
    :param num_bins: int
        A minimum number of bins for the output array

    :return:
        The result of binning the input array
    """

    # for int datatype this is equivalent to the "Convert Bin Centers to Bin Edges" code:
    # https://it.mathworks.com/help/matlab/creating_plots/replace-discouraged-instances-of-hist-and-histc.html#bup38m7-5
    # return all elements but the first one
    return np.bincount(data.flat, minlength=num_bins + 1)[1:]


def s2_multiple_classification(datafile: Path, suffix: str, domain_valid: np.ndarray, domain_solved: np.ndarray,
                               datastack: List[np.ndarray], training_config: Path, classes: List[int],
                               output: Path) -> Path:
    """
    Multiple abstraction class inference

    :param datafile: Path
        the complete filename of the Sentinel 2 data with bands stacked in a vrt
    :param suffix: str
        the letter of the procesing domain: A or B
    :param domain_valid: np.ndarray
        the given domain, A or B
    :param domain_solved: np.ndarray
        the solved domain with minimum support
    :param datastack: List[np.ndarray]
        the quantized datastack
    :param training_config: Path
        the absolute path to the training_config configuration file
    :param classes: List[int]
        the list of classes to extract from data

    :param output: Path
        the complete path where to store classification results
    """

    with rasterio.open(datafile) as src:
        profile = src.profile
        profile.update(
            count=1,
            dtype=np.double,
            driver='GTiff',
            compress='lzw',
            interleave='band',
            blockxsize=256,
            blockysize=256,
            nodata=None,
        )
        crs = src.crs
        bounds = src.bounds
        width = src.width
        height = src.height

    logs('Read CGLS training_config')
    with open(training_config) as tc:
        training = yaml.safe_load(tc)

    classes_flat = []
    layers = []
    for it, cl in enumerate(classes):
        if isinstance(cl, list):
            layer_name = f'{datafile.stem}_dom{suffix}_sml_b{it + 1}_class_{"_".join(str(sc) for sc in cl)}.tif'
            classes_flat.extend(cl)
        else:
            layer_name = f'{datafile.stem}_dom{suffix}_sml_b{it + 1}_class_{cl}.tif'
            classes_flat.append(cl)
        layers.append(output / layer_name)

    for cl in classes_flat:
        if cl not in training['classes']:
            raise KeyError(f'Class {cl} is not a valid class in the configuration file')

    # deal with relative path (from the YAML path location)
    if training['filepath']:
        training_file = Path(training['filepath']) / training['filename']
    else:
        training_file = Path(training['filename'])

    train_cgls = read_ancillary_data(
        filename_ancillary=training_file,
        crs=crs,
        bounds=bounds,
        width=width,
        height=height,
    )

    sigma = 0.5
    trunc = np.ceil(2 * sigma) / sigma

    for cl, layer in zip(classes, layers):
        logs(f'produce inference for abstraction class with codes: {cl}')
        train_mask = np.isin(train_cgls, cl)
        if train_mask.any():
            phi_a, phi_b = sml_minimal_support_multiple_quantization_phi(
                domain_valid=domain_valid,
                domain_solved=domain_solved,
                datastack=datastack,
                training=train_mask,
            )
            phi = (phi_a + phi_b) / 2
        else:
            phi = np.full(shape=domain_valid.shape, fill_value=np.nan)

        sc_phi = gaussian_filter(phi, sigma=sigma, truncate=trunc)

        logs(f'save to file: {layer}')
        with rasterio.open(layer, 'w', **profile) as band:
            band.write(sc_phi, 1)

    datastack_class_file = output / f'{datafile.stem}_dom{suffix}_sml.vrt'
    gdal.BuildVRT(
        destName=str(datastack_class_file),
        srcDSOrSrcDSTab=[str(layer) for layer in layers],
        separate=True, srcNodata=None, VRTNodata=None,
    )

    return datastack_class_file


def sml_minimal_support_multiple_quantization_phi(domain_valid: np.ndarray, domain_solved: np.ndarray,
                                                  datastack: List[np.ndarray],
                                                  training: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Compute phi for multiple quantization values following the hierarchical minimal support map

    :param domain_valid: np.ndarray
        the valid domain where phi value is computed
    :param domain_solved: np.ndarray
        the hierarchical support map for the chosen quantization values
    :param datastack: List[np.ndarray]
        the quantized datastack
    :param training: np.ndarray
        the binary layer used as training_config

    :return: (np.ndarray, np.ndarray)
        the absolute and relative phi results
    """

    # fill the decision map with the phi of the single Q assessment
    phi_global = np.full(shape=training.shape, fill_value=np.nan)
    phi_global_rel = np.full(shape=training.shape, fill_value=np.nan)
    positive_train = np.logical_and(training > 0, domain_valid)
    negative_train = np.logical_and(training == 0, domain_valid)

    for it, data_sequenced in enumerate(datastack, start=1):
        # estimate phi for single level Q
        positive_supp = sml_histogram_count(data=data_sequenced[positive_train], num_bins=data_sequenced.max())
        negative_supp = sml_histogram_count(data=data_sequenced[negative_train], num_bins=data_sequenced.max())

        with np.errstate(invalid='ignore', divide='ignore'):
            phi = (positive_supp - negative_supp) / (positive_supp + negative_supp)
        phi_quant = phi[data_sequenced - 1]

        with np.errstate(invalid='ignore', divide='ignore'):
            positive_supp_rel = positive_supp / positive_train.sum()
            negative_supp_rel = negative_supp / negative_train.sum()
            phi_rel = (positive_supp_rel - negative_supp_rel) / (positive_supp_rel + negative_supp_rel)
        phi_quant_rel = phi_rel[data_sequenced - 1]

        # fill in the multi-Q phi following the hierarchical minimal support map
        processing_domain = domain_solved == it
        phi_global[processing_domain] = phi_quant[processing_domain]
        phi_global_rel[processing_domain] = phi_quant_rel[processing_domain]

    return phi_global, phi_global_rel


def search_maxima(filename: Path, domain_valid: np.ndarray, levels: int, output: Path) -> (Path, Path):
    """
    Search for maximum phi values and their indexes

    :param filename: Path
        the complete filename of the classified datastack
    :param domain_valid: np.ndarray
        the valid domain where phi value is computed
    :param levels: int
        the number of levels used to rescale phi values

    :param output: Path
        the complete path where to write results
    """

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    nodata_mask = ~binary_erosion(domain_valid, selem=kernel)

    with rasterio.open(filename) as src:
        profile = src.profile
        profile.update(
            driver='Gtiff',
            compress='lzw',
            dtype=np.uint8,
            count=1,
        )

        out_class = output / f'{filename.stem}_LC.tif'
        out_class_phi = output / f'{filename.stem}_LC_phi.tif'
        with rasterio.open(out_class, 'w', **profile) as dst_c:
            with rasterio.open(out_class_phi, 'w', **profile) as dst_phi:
                for ji, block in src.block_windows(1):
                    # get data and nodata mask by window block
                    data = src.read(window=block)
                    block_mask = nodata_mask[block.row_off: block.row_off + block.height,
                                             block.col_off: block.col_off + block.width, ]

                    # when all data in a slice are NaN keep the NaN value without rasining a warning
                    warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
                    phi = np.nanmax(data, axis=0)
                    warnings.resetwarnings()
                    # rescale in uint8 while keeping 0 for nodata
                    phi = np.round(imrscl(data=phi, min_value=-1, max_value=1) * (levels - 2)) + 1
                    phi[np.isnan(phi)] = 0
                    phi[block_mask] = 0

                    dst_phi.write(phi.astype(np.uint8), 1, window=block)

                    # set band 1 to value -100 because:
                    # 1. otherwise when all values are NaN we get: "ValueError: All-NaN slice encountered"
                    # 2. to get same behavior as MATLAB max: "If all elements are NaN, then max returns the first one"
                    data[0, np.isnan(data[0, :, :])] = -100
                    # add 1 to have labels starting from 1 instead of 0
                    c = np.nanargmax(data, axis=0) + 1
                    c[block_mask] = 0

                    dst_c.write(c.astype(np.uint8), 1, window=block)

    return out_class, out_class_phi


def generate_class(filesafe: Path, workspace: Path, training: Path, classes: List[int], pixres: int,
                   ) -> (Path, Path, Path, Path):
    """
    Generate classification results using Sentinel 2 bands selected by pixel resolution

    :param filesafe: Path
         the complete filename of the Sentinel 2 SAFE product
    :param workspace: Path
        the absolute path to the working directory
    :param training: Path
        the absolute path to the training_config configuration file
    :param classes: List[int]
        the list of classes to extract from data
    :param pixres: int
        the pixel resolution to filter Sentinel 2 bands, it can be 10 or 20 (meters)

    :return: (Path, Path, Path, Path)
        the filenames of generated classification files:
            - class A
            - class A phi
            - class B
            - class B phi
    """

    logs('Generate class at resolution: {}'.format(pixres))

    logs('Create scratch folder')
    scratch = workspace / 'scratch'
    scratch.mkdir(exist_ok=True)

    logs('Create vrt with pixel resolution: 10m')
    vrt_10m_file = scratch / f'{filesafe.stem}_bands_10m.vrt'
    read_s2_bands_as_vrt(filename_safe=filesafe, pixres=10, out_vrt=vrt_10m_file)

    logs('Split in two domains: A and B')
    # this is always done with the 10m data
    domain_a, domain_b = split_domain(filename=vrt_10m_file, pixres=pixres)

    logs('Create vrt with pixel resolution: 20m')
    vrt_20m_file = scratch / f'{filesafe.stem}_bands_20m.vrt'
    read_s2_bands_as_vrt(filename_safe=filesafe, pixres=20, out_vrt=vrt_20m_file)

    logs('Read vrt data')
    if pixres == 10:
        main_vrt = vrt_10m_file
    else:
        main_vrt = vrt_20m_file

    data_a_file, data_b_file = data_quantile(datafile=main_vrt, domain_a=domain_a, domain_b=domain_b,
                                             nlevels=256, output=scratch)

    class_a_file, class_a_phi_file = process_domain(datafile=main_vrt, suffix='A', dataquant_file=data_a_file,
                                                    domain=domain_a, training=training, classes=classes, output=scratch)

    class_b_file, class_b_phi_file = process_domain(datafile=main_vrt, suffix='B', dataquant_file=data_b_file,
                                                    domain=domain_b, training=training, classes=classes, output=scratch)

    return class_a_file, class_a_phi_file, class_b_file, class_b_phi_file


def process_domain(datafile: Path, suffix: str, dataquant_file: Path, domain: np.ndarray, training: Path,
                   classes: List[int], output: Path) -> (Path, Path):
    """
    Produce classification results for a given domain

    :param datafile: Path
        the complete filename of the Sentinel 2 data with bands stacked in a vrt
    :param suffix: str
        the letter of the procesing domain: A or B
    :param dataquant_file: str
        the complete filename of data quantiled using the given domain
    :param domain: np.ndarray
        the given domain, A or B
    :param training: str
        the absolute path to the training_config configuration file
    :param classes: List[int]
        the list of classes to extract from data

    :param output: Path
        the complete path where to write results
    """

    logs('Process domain: {}'.format(suffix))

    logs('Sequence data encoding minimal-support multiple-quantization')
    # list of quantization values: 1 2 4 8 16 32 64 128
    quantizations = np.power(2, np.arange(8))
    domain_minsupp, datastack_mulquan = sml_minimal_support_multiple_quantization(
        datafile=dataquant_file,
        levels=256,
        minimal_support=100,
        multiple_quantization=quantizations,
    )

    logs('Compute multiple-class multiple-abstraction classification')
    datastack_class_file = s2_multiple_classification(
        datafile=datafile, suffix=suffix, domain_valid=domain, domain_solved=domain_minsupp, datastack=datastack_mulquan,
        training_config=training, classes=classes, output=output)

    logs('Reconciling to a discrete CLASS')
    out_class_file, out_class_phi_file = search_maxima(
        filename=datastack_class_file, domain_valid=domain, levels=256, output=output.parent,
    )

    return out_class_file, out_class_phi_file


def generate_composite(tiffiles: List[str], vrtfile: Path, outfile: Path, outfilephi: Path) -> None:

    with rasterio.open(vrtfile) as vrt:
        dst_bounds = vrt.bounds
        dst_crs = vrt.crs
        dst_height = vrt.height
        dst_width = vrt.width
        profile = vrt.profile

    # Output image transform
    left, bottom, right, top = dst_bounds
    xres = (right - left) / dst_width
    yres = (top - bottom) / dst_height
    dst_transform = affine.Affine(xres, 0.0, left,
                                  0.0, -yres, top)

    vrt_options = {
        'resampling': Resampling.nearest,
        'crs': dst_crs,
        'transform': dst_transform,
        'height': dst_height,
        'width': dst_width,
    }

    # filter out phi files
    files_class = [f for f in tiffiles if 'phi' not in f]

    data = np.zeros(shape=(dst_height, dst_width), dtype=np.uint8)
    data_phi = np.zeros(shape=(dst_height, dst_width), dtype=np.uint8)

    with rasterio.Env(GDAL_CACHEMAX=512):
        for filename in files_class:
            path_phi = filename.replace('.tif', '_phi.tif')

            # find maximum values and indexes for phi values
            with rasterio.open(path_phi) as src_phi:
                with WarpedVRT(src_phi, **vrt_options) as vrt_phi:
                    next_phi = vrt_phi.read(1)
                    better_phi_domain = next_phi > data_phi

                    data_phi[better_phi_domain] = next_phi[better_phi_domain]

            # use max phi indexes to select data
            with rasterio.open(filename) as src:
                with WarpedVRT(src, **vrt_options) as vrt:
                    # Read all data into memory.
                    next_data = vrt.read(1)
                    data[better_phi_domain] = next_data[better_phi_domain]

        # Write output file
        profile.update(
            driver='GTiff',
            compress='lzw',
            interleave='band',
        )

        # write data
        with rasterio.open(outfile, 'w', **profile) as dst:
            dst.write(data, 1)

        # write phi
        with rasterio.open(outfilephi, 'w', **profile) as dst:
            dst.write(data_phi, 1)


def upsampling_20m_to_10m(filename: Path, resampling: str, outfile: Path) -> None:
    gdal.Translate(
        destName=str(outfile),
        srcDS=str(filename),
        xRes=10,
        yRes=10,
        resampleAlg=resampling,
    )


def generate_composite_all(comp10m_data: Path, comp10m_phi: Path, comp20m_data: Path, comp20m_phi: Path,
                           outfile: Path, outfilephi: Path) -> None:
    # read and compare phi
    with rasterio.open(comp10m_phi) as c10m_phi:
        profile = c10m_phi.profile
        x_phi = c10m_phi.read()

    with rasterio.open(comp20m_phi) as c20m_phi:
        next_phi = c20m_phi.read()

    better_phi_domain = next_phi > x_phi
    x_phi[better_phi_domain] = next_phi[better_phi_domain]

    # replace data with phi
    with rasterio.open(comp10m_data) as c10m_data:
        x_data = c10m_data.read()

    with rasterio.open(comp20m_data) as c20m_data:
        next_data = c20m_data.read()

    x_data[better_phi_domain] = next_data[better_phi_domain]

    # Write output file
    with rasterio.open(outfile, 'w', **profile) as c10mALL:
        c10mALL.write(x_data)

    with rasterio.open(outfilephi, 'w', **profile) as c10mALLphi:
        c10mALLphi.write(x_phi)


def generate_composites(files_20m: List[str], files_10m: List[str], output_path: Path) -> None:
    logs('Build VRT 20m')
    # Get AOI bounds
    # create a vrt with all files_10m at 10m
    # TODO: use all files_10m, some files_10m are skipped because of different projections
    vrtfile_20m = output_path / 'S2CG_AOI_20m_py.vrt'
    gdal.BuildVRT(str(vrtfile_20m), files_20m)

    logs('Generate composite 20m')
    composite_20m = output_path / 'composite_S2_CLASS_20m.tif'
    composite_20m_phi = output_path / 'composite_S2_CLASS_20m_phi.tif'
    generate_composite(
        tiffiles=files_20m,
        vrtfile=vrtfile_20m,
        outfile=composite_20m,
        outfilephi=composite_20m_phi,
    )

    logs('Upsample data 20m to 10m')
    composite_20m_to_10m = output_path / 'composite_S2_CLASS_20m_to_10m.tif'
    upsampling_20m_to_10m(
        filename=composite_20m,
        outfile=composite_20m_to_10m,
        resampling='nearest',
    )

    logs('Upsample phi 20m to 10m')
    composite_20m_to_10m_phi = output_path / 'composite_S2_CLASS_20m_to_10m_phi.tif'
    upsampling_20m_to_10m(
        filename=composite_20m_phi,
        outfile=composite_20m_to_10m_phi,
        resampling='bilinear',
    )

    logs('Build VRT 10m')
    vrtfile_10m = output_path / 'S2CG_AOI_10m_py.vrt'
    gdal.BuildVRT(str(vrtfile_10m), files_10m)

    logs('Generate composite 10m')
    composite_10m = output_path / 'composite_S2_CLASS_10m.tif'
    composite_10m_phi = output_path / 'composite_S2_CLASS_10m_phi.tif'
    generate_composite(
        tiffiles=files_10m,
        vrtfile=vrtfile_10m,
        outfile=composite_10m,
        outfilephi=composite_10m_phi,
    )

    logs('Merge in composite ALL')
    composite_all = output_path / 'composite_S2_CLASS_ALL.tif'
    composite_all_phi = output_path / 'composite_S2_CLASS_ALL_phi.tif'
    generate_composite_all(
        comp10m_data=composite_10m,
        comp10m_phi=composite_10m_phi,
        comp20m_data=composite_20m_to_10m,
        comp20m_phi=composite_20m_to_10m_phi,
        outfile=composite_all,
        outfilephi=composite_all_phi,
    )
