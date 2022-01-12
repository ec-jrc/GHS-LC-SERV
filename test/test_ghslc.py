import pytest
import yaml
import requests
import git
from ghslc import ghslc
from pathlib import Path
import hashlib
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.append('..')


def hash_file(filename):
    with open(filename, 'rb') as f:
        file_hash = hashlib.md5()
        for chunk in iter(lambda: f.read(8192), b''):
            file_hash.update(chunk)
    return file_hash.hexdigest()


@pytest.fixture
def cgls_data():
    repo = git.Repo('', search_parent_directories=True)
    repo_root = repo.working_tree_dir
    cgls_config = Path(repo_root) / 'training_CGLS.yml'
    with open(cgls_config) as conf:
        cgls = yaml.safe_load(conf)

    # This is the same file used in training_CGLS.yml
    if cgls['filepath']:
        if cgls['filepath'].startswith('.'):
            cgls_file = cgls_config.parent / cgls['filepath'] / cgls['filename']
        else:
            cgls_file = Path(cgls['filepath']) / cgls['filename']
    else:
        cgls_file = Path(cgls['filename'])

    # download the file if missing
    if not cgls_file.exists():
        cgls_file.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(cgls['url'])
        with open(cgls_file, 'wb') as f:
            f.write(r.content)

    # check integrity using MD5 checksum
    assert hash_file(cgls_file) == '6898fbe3fb46a1110cd65e3a81ed7624'


def test_class_generate(cgls_data):
    repo = git.Repo('', search_parent_directories=True)
    repo_root = repo.working_tree_dir

    training_config = Path(repo_root) / 'training_CGLS.yml'
    classes = [
        80,  # Permanent water bodies
        70,  # Snow and Ice
        60,  # Bare / sparse vegetation
        90,  # Herbaceous wetland
        20,  # Shrubs
        30,  # Herbaceous vegetation
        [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126],  # Forests
        40,  # Cultivated and managed vegetation/agriculture (cropland)
        50,  # Urban / built up
    ]

    # S2 data
    granule = Path(repo_root) / 'test/data/S2A_MSIL1C_20191210T101411_N0208_R022_T32TQM_20191210T104357.SAFE'
    print('S2 data: {}'.format(granule))

    workspace = granule.parent
    print('Workspace: {}'.format(workspace))

    params = [
        (granule, workspace, training_config, classes, 'A', 10),
        (granule, workspace, training_config, classes, 'B', 10),
        (granule, workspace, training_config, classes, 'A', 20),
        (granule, workspace, training_config, classes, 'B', 20),
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = [res for res in executor.map(ghslc.generate_class, *zip(*params))]

    results_10m_class = [item for sublist in results[:2] for item in sublist]
    results_20m_class = [item for sublist in results[2:] for item in sublist]

    assert hash_file(results_10m_class[0]) == '93f4d86d0ef622afde1a9491edf61910'
    assert hash_file(results_10m_class[1]) == '6a306013099a5f39eaa7ad4c0b1cb6e2'
    assert hash_file(results_10m_class[2]) == '236cbf81564ca941ee46660d97c087c5'
    assert hash_file(results_10m_class[3]) == 'f2075dbc7d5aaabf2594299b73b27d87'

    assert hash_file(results_20m_class[0]) == '44c71cd9d41aba6f17ea8071e135a3e1'
    assert hash_file(results_20m_class[1]) == 'fb53ee76799c970676fc38064cbd4212'
    assert hash_file(results_20m_class[2]) == 'b1d4274951c15ee2b6fe9f5bd4529605'
    assert hash_file(results_20m_class[3]) == '437dbef70b2d52bfd37b2ccc9977b5ab'


def test_class_composite():

    workspace = Path('/Volumes/Data/Documents/JRC/workspace/luca.maffenini@ext.ec.europa.eu/d76c42d9-dcb6-4832-8300-1d281983a508')

    files_lc_20m = list(workspace.glob('*bands_20m_dom*.tif'))
    files_lc_10m = list(workspace.glob('*bands_10m_dom*.tif'))

    ghslc.generate_composites(
        files_10m=files_lc_10m,
        files_20m=files_lc_20m,
    )

    assert True
