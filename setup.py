from setuptools import setup, find_namespace_packages

setup(
    name='PokerGTO',
    version='0.1.0',
    packages=find_namespace_packages(where='src', exclude=['tests*']),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=2.1.2',
        'torch>=2.6.0',
        'PyQt6>=6.5.0',
        'matplotlib>=3.7.1',
        'requests>=2.32.3',
        'tqdm>=4.65.0',
        'PyQtWebEngine>=6.5.0',
    ],
    author='Your Name',
    description='Game Theory Optimal Poker Strategy AI',
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'pokergto=src.interface.main_window:main',
        ],
    },
)