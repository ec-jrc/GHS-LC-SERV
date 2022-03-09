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
    check_and_download(s2_10m_file, 'b6cf7d6b27dd4b6c70fcd6cecdb7072e')

    s2_20m_file = repopath / 'test' / 'data' / 'SENTINEL2_L1C_20m.tif'
    check_and_download(s2_20m_file, '41dfa876010f3ec71e02b7ef6612f723')

    return s2_10m_file, s2_20m_file


@pytest.fixture(scope='session')
def s2_data_classified(repopath):
    ref_path = repopath / 'test' / 'data' / 'reference'
    if not ref_path.exists():
        ref_path.mkdir(parents=True, exist_ok=True)

    s2_10m_a_lc = ref_path / 'SENTINEL2_L1C_10m_domA_sml_LC.tif'
    check_and_download(s2_10m_a_lc, 'a9fe361e3446e4ed086cdebd84aa8c54')

    s2_10m_a_phi = ref_path / 'SENTINEL2_L1C_10m_domA_sml_LC_phi.tif'
    check_and_download(s2_10m_a_phi, '09352998746cc9ac8c75ddacf0db453f')

    s2_10m_b_lc = ref_path / 'SENTINEL2_L1C_10m_domB_sml_LC.tif'
    check_and_download(s2_10m_b_lc, '83852e5cbbbe5ce0f8739841ec9f3a78')

    s2_10m_b_phi = ref_path / 'SENTINEL2_L1C_10m_domB_sml_LC_phi.tif'
    check_and_download(s2_10m_b_phi, '2369b619739054d521094134fdba2a86')

    s2_20m_a_lc = ref_path / 'SENTINEL2_L1C_20m_domA_sml_LC.tif'
    check_and_download(s2_20m_a_lc, '11389d555b40b1f1f9bbf66e40e7f336')

    s2_20m_a_phi = ref_path / 'SENTINEL2_L1C_20m_domA_sml_LC_phi.tif'
    check_and_download(s2_20m_a_phi, '47540134866e30727baddbd43abd3a63')

    s2_20m_b_lc = ref_path / 'SENTINEL2_L1C_20m_domB_sml_LC.tif'
    check_and_download(s2_20m_b_lc, 'b6a2a1e66d631ef48653fcf03da4b2bc')

    s2_20m_b_phi = ref_path / 'SENTINEL2_L1C_20m_domB_sml_LC_phi.tif'
    check_and_download(s2_20m_b_phi, '5ff3b036f7169508bc2a465a9b2bfe2f')

    return s2_10m_a_lc, s2_10m_a_phi, s2_10m_b_lc, s2_10m_b_phi, s2_20m_a_lc, s2_20m_a_phi, s2_20m_b_lc, s2_20m_b_phi


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


@pytest.fixture(scope='session')
def artifacts(repopath):
    artifacts_path = repopath / 'test' / 'data' / 'artifacts'
    if not artifacts_path.exists():
        artifacts_path.mkdir(parents=True, exist_ok=True)

    return artifacts_path


def test_class_generate_dom_a_res_10(cgls_data, s2_data, target_classes, artifacts):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=artifacts,
        training=cgls_data,
        classes=target_classes,
        suffix='A',
        pixres=10,
    )
    assert hash_file(res_class) == 'a9fe361e3446e4ed086cdebd84aa8c54'
    assert hash_file(res_phi) == '09352998746cc9ac8c75ddacf0db453f'


def test_class_generate_dom_b_res_10(cgls_data, s2_data, target_classes, artifacts):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=artifacts,
        training=cgls_data,
        classes=target_classes,
        suffix='B',
        pixres=10,
    )
    assert hash_file(res_class) == '83852e5cbbbe5ce0f8739841ec9f3a78'
    assert hash_file(res_phi) == '2369b619739054d521094134fdba2a86'


def test_class_generate_dom_a_res_20(cgls_data, s2_data, target_classes, artifacts):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=artifacts,
        training=cgls_data,
        classes=target_classes,
        suffix='A',
        pixres=20,
    )
    assert hash_file(res_class) == '11389d555b40b1f1f9bbf66e40e7f336'
    assert hash_file(res_phi) == '47540134866e30727baddbd43abd3a63'


def test_class_generate_dom_b_res_20(cgls_data, s2_data, target_classes, artifacts):
    res_class, res_phi = ghslc.generate_class(
        file_10m=s2_data[0],
        file_20m=s2_data[1],
        output=artifacts,
        training=cgls_data,
        classes=target_classes,
        suffix='B',
        pixres=20,
    )
    assert hash_file(res_class) == 'b6a2a1e66d631ef48653fcf03da4b2bc'
    assert hash_file(res_phi) == '5ff3b036f7169508bc2a465a9b2bfe2f'


def test_class_composite_results(s2_data_classified, artifacts):

    (res_comp_20m, res_comp_20m_phi, res_comp_20m_count,
     res_comp_10m, res_comp_10m_phi, res_comp_10m_count,
     res_comp_all, res_comp_all_phi) = ghslc.generate_composites(
        files_10m=s2_data_classified[:4],
        files_20m=s2_data_classified[4:],
        output=artifacts,
    )

    assert hash_file(res_comp_20m) == '337f789006d304eb7cf5453013e7e7b6'
    assert hash_file(res_comp_20m_phi) == '70d2b84792c2de908219e90f8fc611ca'
    assert hash_file(res_comp_20m_count) == '269f524ee6226645593f130d61bca7d0'

    assert hash_file(res_comp_10m) == 'c51da9167732ba210e1c5b4c96c02d93'
    assert hash_file(res_comp_10m_phi) == 'f97ba804963304df3624b1c80fc34d6b'
    assert hash_file(res_comp_10m_count) == 'd4b1b6b9e751c73c3adaa5fac49dc8c0'

    assert hash_file(res_comp_all) == 'a69cf0c90cdd0545a857a98da5408a9d'
    assert hash_file(res_comp_all_phi) == '0987277f42b10ae8a76b9abcecbe969c'
