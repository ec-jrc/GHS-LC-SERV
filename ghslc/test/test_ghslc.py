import pytest
import yaml
import requests
from ghslc import ghslc
from pathlib import Path
import hashlib


def hash_file(filename):
    with open(filename, 'rb') as f:
        file_hash = hashlib.md5()
        for chunk in iter(lambda: f.read(8192), b''):
            file_hash.update(chunk)
    return file_hash.hexdigest()


@pytest.fixture
def cgls_data():
    cgls_config = Path('../../training_CGLS.yml')
    with open(cgls_config) as conf:
        cgls = yaml.safe_load(conf)

    # This is the same file used in training_CGLS.yml
    if cgls['filename'].startswith('.'):
        cgls_file = cgls_config.parent / cgls['filename']
    else:
        cgls_file = Path(cgls['filename'])

    # download the file if missing
    if not cgls_file.exists():
        r = requests.get(cgls['url'])
        with open(cgls_file, 'wb') as f:
            f.write(r.content)

    # check integrity using MD5 checksum
    assert hash_file(cgls_file) == '6898fbe3fb46a1110cd65e3a81ed7624'


def test_class_generate(cgls_data):
    rootpath = Path('./data')

    training_config = Path('../../training_CGLS.yml')
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
    granule = Path(
        'U:/SRS/Copernicus/S2/scenes/source/L1C/2019/12/10/022/S2A_MSIL1C_20191210T101411_N0208_R022_T32TQM_20191210T104357.SAFE'
    )
    print('S2 data: {}'.format(granule))

    workspace_safe = Path(rootpath) / Path(granule).stem
    workspace_safe.mkdir(parents=True, exist_ok=True)
    print('Workspace: {}'.format(workspace_safe))

    results_10m_class = ghslc.generate_class(
        filesafe=granule,
        workspace=workspace_safe,
        training=training_config,
        classes=classes,
        pixres=10,
    )
    assert hash_file(results_10m_class[0]) == 'b7b27c6609657b894224287c1430b798'
    assert hash_file(results_10m_class[1]) == '857e9df4b8f9276c54cc53fb1daeb6d0'
    assert hash_file(results_10m_class[2]) == '6caa7e85a56514d3454717f47d172d1e'
    assert hash_file(results_10m_class[3]) == 'ad1774572dbbecc5f18d45b86aa89401'

    results_20m_class = ghslc.generate_class(
        filesafe=granule,
        workspace=workspace_safe,
        training=training_config,
        classes=classes,
        pixres=20,
    )
    assert hash_file(results_20m_class[0]) == '5e58c55af2d7fdd873c52beaeb4883c9'
    assert hash_file(results_20m_class[1]) == '1d5a966e1a2d9eafaa15b94abc14a5c0'
    assert hash_file(results_20m_class[2]) == '3c223310527f6c3121fb16c2c4901e68'
    assert hash_file(results_20m_class[3]) == 'e02b1be1ef00b74e0d93c5a49b6ea3a4'
