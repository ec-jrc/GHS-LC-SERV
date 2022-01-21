import pytest
import yaml
import git
import requests
from pathlib import Path
import hashlib
from ghslc import ghslc


def hash_file(filename):
    with open(filename, 'rb') as f:
        file_hash = hashlib.md5()
        for chunk in iter(lambda: f.read(8192), b''):
            file_hash.update(chunk)
    return file_hash.hexdigest()


@pytest.fixture
def repopath():
    repo = git.Repo('', search_parent_directories=True)
    repo_root = repo.working_tree_dir
    return Path(repo_root)


@pytest.fixture
def cgls_data(repopath):

    # Get CGLS raster
    cgls_config = repopath / 'training_CGLS.yml'
    with open(cgls_config) as conf:
        cgls = yaml.safe_load(conf)

    # Read and adapt filename and relative path
    if cgls['filepath']:
        if cgls['filepath'].startswith('.'):
            cgls_file = cgls_config.parent / cgls['filepath'] / cgls['filename']
        else:
            cgls_file = Path(cgls['filepath']) / cgls['filename']
    else:
        cgls_file = Path(cgls['filename'])

    # download the file if missing
    print(f'Using CGLS file: {cgls_file}')
    if not cgls_file.exists():
        print('CGLS file not found! Downloading ...')
        cgls_file.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(cgls['url'])
        print(f'CGLS file downloaded. Saving to {cgls_file}')
        with open(cgls_file, 'wb') as f:
            f.write(r.content)

    # check integrity using MD5 checksum
    print('Check CGLS file MD5 checksum')
    assert hash_file(cgls_file) == '6898fbe3fb46a1110cd65e3a81ed7624'

    return cgls_config


@pytest.fixture
def s2_data(repopath):
    s2_file = repopath / 'test' / 'data' / 'S2A_MSIL1C_20191210T101411_N0208_R022_T32TQM_20191210T104357.zip'
    print(f'Using S2 file: {s2_file}')

    if not s2_file.exists():
        print('S2 file not found! Downloading ...')
        r = requests.get(f'https://ghslsys.jrc.ec.europa.eu/download/ghslc_test_data/{s2_file.name}')
        print(f'S2 file downloaded. Saving to {s2_file}')
        with open(s2_file, 'wb') as f:
            f.write(r.content)

    return s2_file


@pytest.fixture
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
    res_class, res_phi = ghslc.generate_class(s2_data, s2_data.parent, cgls_data, target_classes, 'A', 10)
    assert hash_file(res_class) == '1b72a5a5d4d9cd8593688927a4b507cf'
    assert hash_file(res_phi) == '528bea4e04d8b33fa289aa938c37c75f'


def test_class_generate_dom_b_res_10(cgls_data, s2_data, target_classes):
    res_class, res_phi = ghslc.generate_class(s2_data, s2_data.parent, cgls_data, target_classes, 'B', 10)
    assert hash_file(res_class) == 'ab45e538957cdc5487589d7aaa9ba5ce'
    assert hash_file(res_phi) == '29cf71f47f83c208801bf84a1ad39842'


def test_class_generate_dom_a_res_20(cgls_data, s2_data, target_classes):
    res_class, res_phi = ghslc.generate_class(s2_data, s2_data.parent, cgls_data, target_classes, 'A', 20)
    assert hash_file(res_class) == 'ae19f4181462067ee4f2006510136e9d'
    assert hash_file(res_phi) == 'ff9033a5f62db960e0f492bb3614e633'


def test_class_generate_dom_b_res_20(cgls_data, s2_data, target_classes):
    res_class, res_phi = ghslc.generate_class(s2_data, s2_data.parent, cgls_data, target_classes, 'B', 20)
    assert hash_file(res_class) == '95f875dd8c49d02e3177095fde629fe3'
    assert hash_file(res_phi) == '342e1f07cf79e1dfecba155acda53165'
