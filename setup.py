"""Hartonomous setup"""
from setuptools import setup, find_packages

setup(
    name='hartonomous',
    version='0.3.0',
    description='PostgreSQL/PostGIS as Geometric AI Substrate',
    author='aharttn',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'psycopg2-binary>=2.9',
        'numpy>=1.24',
        'safetensors>=0.4',
        'tqdm>=4.65',
    ],
    entry_points={
        'console_scripts': [
            'hartonomous=hartonomous.cli.__main__:main',
        ],
    },
    python_requires='>=3.8',
)
