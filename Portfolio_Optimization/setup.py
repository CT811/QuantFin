from setuptools import setup

VERSION = '0.0.1'
DESCRIPTION = 'Calculate optimized portfolio returns and compare them with SP500'

# Setting up
setup(
    name="PortOptPy",
    version=VERSION,
    author="Chrysafis Tsoukalas",
    description=DESCRIPTION,
    install_requires=['numpy', 
                      'pandas',
                      'yfinance',
                      'random',
                      'scipy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)