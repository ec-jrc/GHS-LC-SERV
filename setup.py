import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ghslc',
    version='0.0.1',
    author='Luca Maffenini',
    author_email='jrc-ghsl-tools@ec.europa.eu',
    description='GHSL Land Cover python module',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ec-jrc/GHS-LC-SERV',
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'gitpython'],
    # license='GPL v3',
    # packages=['toolbox'],
    # install_requires=['requests'],
)
