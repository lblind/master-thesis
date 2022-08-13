from setuptools import setup

setup(
    name='master-thesis',
    version='1.1',
    packages=['master-thesis'],
    url='',
    license='',
    author='L. Blind',
    author_email='l.blind@student.maastrichtuniversity.nl',
    description='Implementation of Master Thesis Project',
    install_requires=['numpy', 'pandas', 'matplotlib', 'seaborn', 'openpyxl', 'xarray', 'geopy', 'missingno',
                      'geopandas', 'pydmd']
)
