import git
import yaml
import numpy as np
import pytest
import rasterio
import requests

from pathlib import Path
import hashlib

from ghslc import ghslc


def check_and_download(filename, md5):
    # check data file exists, otherwise download
    print(f'Checking file: {filename}')
    if not filename.exists():
        print('File not found! Downloading ...')
        filename.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f'https://ghslsys.jrc.ec.europa.eu/download/ghslc_test_data/{filename.name}')

        print(f'File downloaded. Saving to {filename}')
        with open(filename, 'wb') as f:
            f.write(r.content)

    # check integrity using MD5 checksum
    print('Check file MD5 checksum')
    assert hash_file(filename) == md5

    print('Checksum OK')
    return filename


def hash_file(filename):
    with open(filename, 'rb') as f:
        file_hash = hashlib.md5()
        for chunk in iter(lambda: f.read(4096), b''):
            file_hash.update(chunk)
    return file_hash.hexdigest()


@pytest.fixture(scope='session')
def repopath():
    repo = git.Repo('', search_parent_directories=True)
    repo_root = repo.working_tree_dir
    return Path(repo_root)


@pytest.fixture(scope='session')
def cgls_data(repopath):
    # Get CGLS raster
    cgls_config_file = repopath / 'training_CGLS.yml'
    with open(cgls_config_file) as conf:
        cgls = yaml.safe_load(conf)

    # Read and adapt filename and relative path
    if cgls['filepath']:
        if cgls['filepath'].startswith('.'):
            cgls_file = cgls_config_file.parent / cgls['filepath'] / cgls['filename']
        else:
            cgls_file = Path(cgls['filepath']) / cgls['filename']
    else:
        cgls_file = Path(cgls['filename'])

    check_and_download(cgls_file, cgls['md5'])

    return cgls_config_file


@pytest.fixture(scope='session')
def s2_data(repopath):
    s2_10m_file = repopath / 'test' / 'data' / 'SENTINEL2_L1C_10m.tif'
    check_and_download(s2_10m_file, 'e53a83c20d42d9cbebc349b830175df0')

    s2_20m_file = repopath / 'test' / 'data' / 'SENTINEL2_L1C_20m.tif'
    check_and_download(s2_20m_file, '95c328b4a8500e26226750ae88764a01')

    return dict(s2_10m=s2_10m_file, s2_20m=s2_20m_file)


@pytest.fixture(scope='session')
def sml_data():
    x = np.array(
        [[[202, 10, 44, 117],
          [1, 44, 231, 165],
          [250, 176, 48, 73]],
         [[194, 200, 161, 61],
          [36, 188, 29, 11],
          [121, 69, 21, 154]]]
    )

    seq_ref = np.array(
        [[49866, 51210, 41260, 15733],
         [9217, 48172, 7655, 2981],
         [31226, 17840, 5424, 39497]],
        dtype=int,
    )

    return dict(x=x, seq_ref=seq_ref)


@pytest.fixture(scope='session')
def target_classes():
    classes = [
        [80, 200],  # Permanent water bodies
        [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126],  # Forests
        [20, 30, 40, 60, 90],  # Vegetation / agriculture
        50,  # Urban / built up
    ]
    return classes


def test_threshold_otsu_is_the_same_as_matlab(s2_data):
    with rasterio.open(s2_data['s2_10m']) as src:
        image = src.read()
    thr_otsu = ghslc.threshold_otsu(image)

    # same result as from MATLAB's multithresh function
    assert thr_otsu == 1647


def test_domain_a_10m_covers_most_of_pixels(s2_data):
    domain_a = ghslc.split_domain(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        suffix='A',
        pixres=10,
    )

    # most pixels are in domain A
    assert domain_a.sum() > 5800000

    # there is a tiny hole outside the domain
    assert np.all(np.unique(domain_a[1785:1790, 1895:1900]) == np.array(False))

    # a big no domain area overalaps the nodata area in the lower right corner
    assert np.all(np.unique(domain_a[1800:-1, 2800:-1]) == np.array(False))


def test_domain_b_10m_is_the_opposite_of_a_but_nodata(s2_data):
    domain_b = ghslc.split_domain(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        suffix='B',
        pixres=10,
    )

    # almost no pixel in domain B
    assert domain_b.sum() < 200

    # only a tiny hole is in domain B
    assert np.all(np.unique(domain_b[1785:1790, 1895:1900]) == np.array(True))

    # the same nodata area in the lower right corner stays nodata
    assert np.all(np.unique(domain_b[1800:-1, 2800:-1]) == np.array(False))


def test_domain_20m_is_the_same_but_smaller(s2_data):
    domain_b = ghslc.split_domain(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        suffix='B',
        pixres=20,
    )

    # domain is the same but half the size
    assert np.all(np.unique(domain_b[1785 // 2:1790 // 2, 1895 // 2:1900 // 2]) == np.array(True))


def test_data_quantile_output_expected_dimensions_and_datatype(s2_data, tmp_path):
    domain_a = ghslc.split_domain(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        suffix='A',
        pixres=10,
    )

    dataquant_file = ghslc.data_quantile(
        datafile=s2_data['s2_10m'],
        suffix='A',
        domain=domain_a,
        nlevels=256,
        output=tmp_path,
    )

    with rasterio.open(dataquant_file) as res:
        data = res.read()

    # the 10m input produce a 4 bands results
    assert data.shape[0] == 4

    # all values are quantized in 256 levels
    assert np.all(np.isin(np.unique(data), np.arange(255)))


def test_sml_sequence_encode_throw_error_if_num_level_greater_then_base_input(sml_data):
    with pytest.raises(ValueError):
        ghslc.sml_sequence_encode(sml_data['x'], 10)


def test_sml_sequence_encode_output_equal_legacy_results(sml_data):
    seq_enc = ghslc.sml_sequence_encode(sml_data['x'], 256)

    assert np.array_equal(seq_enc, sml_data['seq_ref'])


def test_sml_histogram_count_is_the_same_as_matlab():
    x = np.array([-9, -6, -5, -2, 0, 1, 3, 3, 4, 7]) + 9

    # output of MATLAB's histcounts
    ref = np.array([0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 0, 0, 1])

    counts = ghslc.sml_histogram_count(x, 4)

    assert np.array_equal(counts, ref)


def test_sml_minimal_support_multiple_quantization_output_expected_datastack(s2_data, tmp_path):
    domain_a = ghslc.split_domain(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        suffix='A',
        pixres=10,
    )

    dataquant_file = ghslc.data_quantile(
        datafile=s2_data['s2_10m'],
        suffix='A',
        domain=domain_a,
        nlevels=256,
        output=tmp_path,
    )

    quantizations = [1, 2, 4, 8, 16, 32, 64, 128]
    domain_minsupp, datastack_mulquan = ghslc.sml_minimal_support_multiple_quantization(
        datafile=dataquant_file,
        levels=256,
        minimal_support=100,
        multiple_quantization=quantizations,
    )

    # the domain values goes from 1 to the length of quantization list
    assert np.all(np.isin(np.unique(domain_minsupp), np.arange(start=1, stop=len(quantizations) + 1)))

    # the datastack is a list with an element for each quantization value
    assert len(datastack_mulquan) == len(quantizations)

    # each element of the datastack is raster with same shape as the input data (check only the first)
    assert datastack_mulquan[0].shape == domain_a.shape


def test_sml_minimal_support_multiple_quantization_phi_output_is_classified(s2_data, tmp_path):
    domain_a = ghslc.split_domain(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        suffix='A',
        pixres=10,
    )

    dataquant_file = ghslc.data_quantile(
        datafile=s2_data['s2_10m'],
        suffix='A',
        domain=domain_a,
        nlevels=256,
        output=tmp_path,
    )

    quantizations = [1, 2, 4, 8, 16, 32, 64, 128]
    domain_minsupp, datastack_mulquan = ghslc.sml_minimal_support_multiple_quantization(
        datafile=dataquant_file,
        levels=256,
        minimal_support=100,
        multiple_quantization=quantizations,
    )

    # simulate water training
    train_mask = np.zeros_like(domain_a)
    train_mask[1650:-1, :500] = True
    train_mask[:250, 1890:2400] = True
    train_mask[:100, 2780:2870] = True

    phi_a, phi_b = ghslc.sml_minimal_support_multiple_quantization_phi(
        domain_valid=domain_a,
        domain_solved=domain_minsupp,
        datastack=datastack_mulquan,
        training=train_mask,
    )

    # positively classified pixels are way more than negative in the positive training area
    assert np.sum(phi_a[1650:-1, :500] >= 0) / np.sum(phi_a[1650:-1, :500] < 0) > 10
    assert np.sum(phi_a[:250, 1890:2400] >= 0) / np.sum(phi_a[:250, 1890:2400] < 0) > 3

    assert np.sum(phi_b[1650:-1, :500] >= 0) / np.sum(phi_b[1650:-1, :500] < 0) > 100
    assert np.sum(phi_b[:250, 1890:2400] >= 0) / np.sum(phi_b[:250, 1890:2400] < 0) > 100


def test_s2_multiple_classification_output_contains_target_classes(s2_data, cgls_data, target_classes, tmp_path):
    domain_a = ghslc.split_domain(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        suffix='A',
        pixres=10,
    )

    dataquant_file = ghslc.data_quantile(
        datafile=s2_data['s2_10m'],
        suffix='A',
        domain=domain_a,
        nlevels=256,
        output=tmp_path,
    )

    quantizations = [1, 2, 4, 8, 16, 32, 64, 128]
    domain_minsupp, datastack_mulquan = ghslc.sml_minimal_support_multiple_quantization(
        datafile=dataquant_file,
        levels=256,
        minimal_support=100,
        multiple_quantization=quantizations,
    )

    datastack_class_file = ghslc.s2_multiple_classification(
        datafile=s2_data['s2_10m'],
        suffix='A',
        domain_valid=domain_a,
        domain_solved=domain_minsupp,
        datastack=datastack_mulquan,
        training_config=cgls_data,
        classes=target_classes,
        output=tmp_path,
    )

    with rasterio.open(datastack_class_file) as res:
        bands = res.count
        vrt_files = res.files

    # there is one band for each target class group
    assert bands == len(target_classes)

    for file in vrt_files:
        assert Path(file).exists()


def test_generate_class_creates_both_landcover_and_phi_output(s2_data, cgls_data, target_classes, tmp_path):
    cl_a_10m_file, phi_a_10m_file = ghslc.generate_class(
        file_10m=s2_data['s2_10m'],
        file_20m=s2_data['s2_20m'],
        output=tmp_path,
        training=cgls_data,
        classes=target_classes,
        minimal_support=100,
        suffix='A',
        pixres=10,
    )

    with rasterio.open(cl_a_10m_file) as cls:
        lc_cls = cls.read()

    # the new land cover classes are a completely contained in the target classes
    assert np.all(np.isin(np.unique(lc_cls), np.arange(len(target_classes) + 1)))

    # the bottom left is correctly classified as class 1 (water)
    assert np.all(lc_cls[1650:-1, :500] == 1)

    assert phi_a_10m_file.exists()
