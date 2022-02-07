import pytest
import yaml
import git
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
        for chunk in iter(lambda: f.read(8192), b''):
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
    check_and_download(s2_10m_file, 'b6cf7d6b27dd4b6c70fcd6cecdb7072e')

    s2_20m_file = repopath / 'test' / 'data' / 'SENTINEL2_L1C_20m.tif'
    check_and_download(s2_20m_file, '41dfa876010f3ec71e02b7ef6612f723')

    return s2_10m_file, s2_20m_file


@pytest.fixture(scope='session')
def target_classes():
    classes = [
        [80, 200],  # Permanent water bodies
        70,  # Snow and Ice
        60,  # Bare / sparse vegetation
        90,  # Herbaceous wetland
        20,  # Shrubs
        30,  # Herbaceous vegetation
        [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126],  # Forests
        40,  # Cultivated and managed vegetation/agriculture (cropland)
        50,  # Urban / built up
    ]
    return classes


def test_class_generate_dom_a_res_10(cgls_data, s2_data, target_classes):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=s2_data[0].parent,
        training=cgls_data,
        classes=target_classes,
        suffix='A',
        pixres=10,
    )
    assert hash_file(res_class) == 'a9fe361e3446e4ed086cdebd84aa8c54'
    assert hash_file(res_phi) == '09352998746cc9ac8c75ddacf0db453f'


def test_class_generate_dom_b_res_10(cgls_data, s2_data, target_classes):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=s2_data[0].parent,
        training=cgls_data,
        classes=target_classes,
        suffix='B',
        pixres=10,
    )
    assert hash_file(res_class) == '83852e5cbbbe5ce0f8739841ec9f3a78'
    assert hash_file(res_phi) == '2369b619739054d521094134fdba2a86'


def test_class_generate_dom_a_res_20(cgls_data, s2_data, target_classes):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=s2_data[0].parent,
        training=cgls_data,
        classes=target_classes,
        suffix='A',
        pixres=20,
    )
    assert hash_file(res_class) == '11389d555b40b1f1f9bbf66e40e7f336'
    assert hash_file(res_phi) == '47540134866e30727baddbd43abd3a63'


def test_class_generate_dom_b_res_20(cgls_data, s2_data, target_classes):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=s2_data[0].parent,
        training=cgls_data,
        classes=target_classes,
        suffix='B',
        pixres=20,
    )
    assert hash_file(res_class) == 'b6a2a1e66d631ef48653fcf03da4b2bc'
    assert hash_file(res_phi) == '5ff3b036f7169508bc2a465a9b2bfe2f'
