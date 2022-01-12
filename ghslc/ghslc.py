from typing import Iterable, List
from pathlib import Path
import zipfile
from osgeo import gdal
import rasterio
from rasterio.profiles import DefaultGTiffProfile
from rasterio.coords import BoundingBox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from affine import Affine
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


def generate_classification(filesafe: Path, workspace: Path, training: Path, classes: List[int]) -> Iterable[Path]:
    """
    Generate classification results at 10 and 20 meters for both domains A and B

    :param filesafe: Path
        the complete filename of the Sentinel 2 data. It can be a .SAFE folder or a zip file
    :param workspace: Path
        the complete path where to store results
    :param training: Path
        the absolute path to the training_config configuration file
    :param classes: List[int]
        the list of classes to extract from data

    :return: Iterable[Path]
        the complete path to all classified results saved on disk
    """

    cl_a_10m_file, phi_a_10m_file = generate_class(filesafe=filesafe, workspace=workspace, training=training,
                                                   classes=classes, suffix='A', pixres=10)

    cl_b_10m_file, phi_b_10m_file = generate_class(filesafe=filesafe, workspace=workspace, training=training,
                                                   classes=classes, suffix='B', pixres=10)

    cl_a_20m_file, phi_a_20m_file = generate_class(filesafe=filesafe, workspace=workspace, training=training,
                                                   classes=classes, suffix='A', pixres=20)

    cl_b_20m_file, phi_b_20m_file = generate_class(filesafe=filesafe, workspace=workspace, training=training,
                                                   classes=classes, suffix='B', pixres=20)

    return (cl_a_10m_file, phi_a_10m_file, cl_b_10m_file, phi_b_10m_file,
            cl_a_20m_file, phi_a_20m_file, cl_b_20m_file, phi_b_20m_file)


def generate_class(filesafe: Path, workspace: Path, training: Path, classes: List[int], suffix: str, pixres: int,
                   ) -> (Path, Path):
    """

    :param filesafe: Path
        the complete filename of the Sentinel 2 data. It can be a .SAFE folder or a zip file
    :param workspace: Path
        the complete path where to store results
    :param training: Path
        the absolute path to the training_config configuration file
    :param classes: List[int]
        the list of classes to extract from data
    :param suffix: str
        the letter of the processing domain: A or B
    :param pixres: int
        the pixel resolution to filter Sentinel 2 bands, it can be 10 or 20 (meters)

    :return: (Path, Path)
        the complete path to the classified file and the phi value file
    """

    logs('Create scratch folder')
    scratch = workspace / f'{filesafe.stem}_scratch' / f'{pixres}m_{suffix}'
    scratch.mkdir(parents=True, exist_ok=True)

    logs('Create vrt with pixel resolution: 10m')
    vrt_10m_file = scratch / f'{filesafe.stem}_bands_10m.vrt'
    read_s2_bands_as_vrt(filename_safe=filesafe, pixres=10, out_vrt=vrt_10m_file)

    logs('Create vrt with pixel resolution: 20m')
    vrt_20m_file = scratch / f'{filesafe.stem}_bands_20m.vrt'
    read_s2_bands_as_vrt(filename_safe=filesafe, pixres=20, out_vrt=vrt_20m_file)

    logs(f'Split in two domains and get domain: {suffix}')
    # this is always done with the 10m data
    domain = split_domain(filename=vrt_10m_file, suffix=suffix, pixres=pixres)

    if pixres == 10:
        main_vrt = vrt_10m_file
    else:
        main_vrt = vrt_20m_file

    class_file, class_phi_file = process_domain(datafile=main_vrt, suffix=suffix, domain=domain,
                                                training=training, classes=classes, output=scratch)

    return class_file, class_phi_file


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
            jp2s_all = [
                f'/vsizip/{filename_safe}/{zip_file}' for zip_file in zf.namelist() if zip_file.endswith('.jp2')
            ]
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


def split_domain(filename: Path, suffix: str, pixres: int) -> np.ndarray:
    """
    Split the data in two domains A and B based on luminance

    :param filename: Path
        the filename of the dataset composed by S2 10m bands
    :param suffix: str
        the letter of the processing domain: A or B
    :param pixres: int
        the pixel resolution to filter Sentinel 2 bands, it can be 10 or 20 (meters)

    :return: np.ndarray
        the chosen domain A or B
    """

    with rasterio.open(filename) as vrt:
        data = vrt.read()

    # data domain
    domain = data.min(axis=0) > 0
    # luminance
    luminance = data[:3, :, :].max(axis=0)
    # automatic threshold (OTSU)
    thr_otsu = threshold_otsu(luminance[domain])

    if suffix == 'A':
        domain = np.logical_and(domain, luminance <= thr_otsu)
    else:
        domain = np.logical_and(domain, luminance > thr_otsu)

    if pixres == 20:
        shape_20m = [int(ds / 2) for ds in domain.shape]
        domain = np.array(Image.fromarray(domain).resize(shape_20m, Image.NEAREST))

    return domain


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


def process_domain(datafile: Path, suffix: str, domain: np.ndarray, training: Path, classes: List[int],
                   output: Path) -> (Path, Path):
    """
    Produce classification results for a given domain

    :param datafile: Path
        the complete filename of the Sentinel 2 data with bands stacked in a vrt
    :param suffix: str
        the letter of the procesing domain: A or B
    :param domain: np.ndarray
        the given domain, A or B
    :param training: str
        the absolute path to the training_config configuration file
    :param classes: List[int]
        the list of classes to extract from data

    :param output: Path
        the complete path where to write results

    :return: (Path, Path)
        the complete path to the classified file and the phi value file
    """

    logs('Process domain: {}'.format(suffix))

    dataquant_file = data_quantile(datafile=datafile, suffix=suffix, domain=domain, nlevels=256, output=output)

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
        datafile=datafile, suffix=suffix, domain_valid=domain, domain_solved=domain_minsupp,
        datastack=datastack_mulquan, training_config=training, classes=classes, output=output)

    logs('Reconciling to a discrete CLASS')
    out_class_file, out_class_phi_file = search_maxima(
        filename=datastack_class_file, domain_valid=domain, levels=256, output=output.parent,
    )

    return out_class_file, out_class_phi_file


def data_quantile(datafile: Path, suffix: str, domain: np.ndarray, nlevels: int, output: Path) -> Path:
    """
    Quantile data in two domains with in a given number of levels

    :param datafile: Path
        the complete filename of the dataset to quantile
    :param suffix: str
        the letter of the procesing domain: A or B
    :param domain: np.ndarray
        the domain A
    :param nlevels: int
        the number of levels used to quantile

    :param output: Path
        the complete path where to write results

    :return: Path
        the complete path to the quantiled data file
    """

    logs('Quantile data with levels: {}'.format(nlevels))

    saturation = 0.0001
    low_q = saturation
    high_q = 1 - saturation

    with rasterio.open(datafile) as src:
        profile = src.profile.copy()
        profile.update(
            driver='GTiff',
            compress='lzw',
            dtype=np.uint8,
            blockxsize=256,
            blockysize=256,
            interleave='band',
        )

        dst_file = output / f'{datafile.stem}_dom{suffix}_quant.tif'
        logs(f'Save to file: {dst_file}')
        with rasterio.open(dst_file, 'w', **profile) as dst:
            for i in src.indexes:
                logs(f'quantile band: {i}')
                band = src.read(i)

                q_min = np.quantile(band[domain], low_q, interpolation='lower')
                q_max = np.quantile(band[domain], high_q, interpolation='higher')
                x_norm = np.round(imrscl(data=band, min_value=q_min, max_value=q_max) * (nlevels - 2))

                dst.write(x_norm, i)

    return dst_file


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
        profile = src.profile.copy()
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

    if training['filepath']:
        if training['filepath'].startswith('.'):
            # build relative path from YAML file location
            training_file = training_config.parent / training['filepath'] / training['filename']
        else:
            # build absolute path
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

    # Output image transform
    left, bottom, right, top = bounds
    xres = (right - left) / width
    yres = (top - bottom) / height
    dst_transform = Affine(xres, 0.0, left,
                           0.0, -yres, top)

    vrt_options = {
        'resampling': Resampling.nearest,
        'crs': crs,
        'transform': dst_transform,
        'height': height,
        'width': width,
    }

    with rasterio.open(filename_ancillary) as src:
        with WarpedVRT(src, **vrt_options) as vrt:
            data = vrt.read(1)

    return data


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

    :return: (Path, Path)
        the complete path to the classified file and the phi value file
    """

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    nodata_mask = ~binary_erosion(domain_valid, selem=kernel)

    with rasterio.open(filename) as src:
        profile = src.profile.copy()
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


def generate_composites(files_20m: List[Path], files_10m: List[Path]) -> (Path, Path, Path, Path, Path, Path):

    # Get AOI bounds
    bounds = common_extent_mollweide(files_20m=files_20m)

    logs('Generate composite 20m')
    composite_20m, composite_20m_phi = generate_composite(
        tiffiles=files_20m,
        pixres=20,
        bounds=bounds,
    )

    logs('Upsample data 20m to 10m')
    composite_20m_to_10m = upsampling_20m_to_10m(
        filename=composite_20m,
        resampling='nearest',
    )

    logs('Upsample phi 20m to 10m')
    composite_20m_to_10m_phi = upsampling_20m_to_10m(
        filename=composite_20m_phi,
        resampling='nearest',
    )

    # get bounds of data resampled from 20m to 10m as they might be slightly larger due to resampling
    with rasterio.open(composite_20m_to_10m) as src:
        bounds_20m = src.bounds

    logs('Generate composite 10m')
    composite_10m, composite_10m_phi = generate_composite(
        tiffiles=files_10m,
        pixres=10,
        bounds=bounds_20m,
    )

    logs('Merge in composite ALL')
    composite_all, composite_all_phi = generate_composite_all(
        comp10m_data=composite_10m,
        comp10m_phi=composite_10m_phi,
        comp20m_data=composite_20m_to_10m,
        comp20m_phi=composite_20m_to_10m_phi,
    )

    return (composite_20m, composite_20m_phi,
            composite_10m, composite_10m_phi,
            composite_all, composite_all_phi)


def common_extent_mollweide(files_20m: List[Path]):

    # restrict the files list to the one with different S2 product names
    files_20m_unique_s2_names = []
    s2_names = set()
    for file in files_20m:
        product_name = "_".join(file.stem.split('_')[:7])
        if product_name not in s2_names:
            s2_names.add(product_name)
            files_20m_unique_s2_names.append(file)

    # extract bbox from files
    bounds_mw = []
    for file in files_20m_unique_s2_names:
        with rasterio.open(file) as src:
            with WarpedVRT(src, crs='ESRI:54009') as vrt:
                bounds_mw.append(vrt.bounds)

    # compute envelope of all bbox
    envelope = BoundingBox(
        left=min([b.left for b in bounds_mw]),
        bottom=min([b.bottom for b in bounds_mw]),
        right=max([b.right for b in bounds_mw]),
        top=max([b.top for b in bounds_mw]),
    )

    return envelope


def generate_composite(tiffiles: List[Path], pixres: int, bounds: BoundingBox) -> (Path, Path):

    # Output image transform
    left, bottom, right, top = bounds
    width = round((right - left) / pixres)
    height = round((top - bottom) / pixres)
    transform = Affine(pixres, 0.0, left,
                       0.0, -pixres, top)

    vrt_options = {
        'crs': 'ESRI:54009',
        'resampling': Resampling.nearest,
        'transform': transform,
        'height': height,
        'width': width,
    }

    # init data
    data = np.zeros(shape=(height, width), dtype=np.uint8)
    data_phi = np.zeros(shape=(height, width), dtype=np.uint8)

    # filter class and phi files
    files_phi = [f for f in tiffiles if '_phi' in str(f)]
    files_class = [f.parent / f.name.replace('_phi', '') for f in files_phi]

    with rasterio.Env(GDAL_CACHEMAX=512):
        for fileclass, filephi in zip(files_class, files_phi):

            # find maximum values and indexes for phi values
            with rasterio.open(filephi) as src_phi:
                with WarpedVRT(src_phi, **vrt_options) as vrt_phi:
                    next_phi = vrt_phi.read(1)
                    better_phi_domain = next_phi > data_phi

                    data_phi[better_phi_domain] = next_phi[better_phi_domain]

            # use max phi indexes to select data
            with rasterio.open(fileclass) as src:
                with WarpedVRT(src, **vrt_options) as vrt:
                    # Read all data into memory.
                    next_data = vrt.read(1)
                    data[better_phi_domain] = next_data[better_phi_domain]

        # Write output file
        profile = DefaultGTiffProfile(
            count=1,
            width=width,
            height=height,
            crs='ESRI:54009',
            transform=transform,
        )

        # write data
        composite_data = tiffiles[0].parent / f'composite_S2_CLASS_{pixres}m.tif'
        with rasterio.open(composite_data, 'w', **profile) as dst:
            dst.write(data, 1)

        # write phi
        composite_phi = tiffiles[0].parent / f'composite_S2_CLASS_{pixres}m_phi.tif'
        with rasterio.open(composite_phi, 'w', **profile) as dst:
            dst.write(data_phi, 1)

    return composite_data, composite_phi


def upsampling_20m_to_10m(filename: Path, resampling: str) -> Path:
    outfile = filename.parent / filename.name.replace('20m', '20m_to_10m')
    gdal.Translate(
        destName=str(outfile),
        srcDS=str(filename),
        xRes=10,
        yRes=10,
        resampleAlg=resampling,
    )

    return outfile


def generate_composite_all(comp10m_data: Path, comp10m_phi: Path, comp20m_data: Path, comp20m_phi: Path) -> (Path, Path):
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
    composite_all = comp10m_data.parent / comp10m_data.name.replace('10m', 'ALL')
    with rasterio.open(composite_all, 'w', **profile) as c10mALL:
        c10mALL.write(x_data)

    composite_all_phi = comp10m_phi.parent / comp10m_phi.name.replace('10m', 'ALL')
    with rasterio.open(composite_all_phi, 'w', **profile) as c10mALLphi:
        c10mALLphi.write(x_phi)

    return composite_all, composite_all_phi
