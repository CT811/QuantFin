from setuptools import setup

VERSION = '0.0.1'
DESCRIPTION = 'Calculate European option prices using binomial trees'

# Setting up
setup(
    name="bintreeOpt",
    version=VERSION,
    author="Chrysafis Tsoukalas",
    description=DESCRIPTION,
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)