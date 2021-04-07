#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='timESD',
    version='0.0.1',
    description='Time Series Forecasting for Energy Sorage Deployment',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/thbuerg/timESD',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

