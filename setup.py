from setuptools import setup, find_packages


setup(
    name='margipose',
    version='0.1.0',
    author='Aiden Nibali',
    license='Apache Software License 2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'margipose = margipose.bin.__init__:main',
        ],
    }
)
