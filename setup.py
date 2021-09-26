from setuptools import find_packages, setup

setup(
    name="transformer_timeseries",
    version="0.0.1",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    description="QData Research Library for the Spacetimeformer project",
    author="Jake Grigsby",
    author_email="jcg6dn@virginia.edu",
    license="MIT",
    packages=find_packages(),
)
