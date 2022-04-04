from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as readme:
    long_description = readme.read()

setup(
    name='ghslc',
    version='0.1.10',
    author='Luca Maffenini',
    author_email='jrc-ghsl-tools@ec.europa.eu',
    description='GHSL Land Cover python module',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ec-jrc/GHS-LC-SERV',
    packages=find_packages(
        include=['ghslc', 'ghslc.*']
    ),
    python_requires='>=3.7',
    install_requires=[
        'PyYAML',
        'rasterio',
        'GDAL',
        'numpy',
        'pillow',
        'scipy',
        'scikit-image',
    ],
    license='GPLv3',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Image Processing',
    ]
)
