#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='spike',
    version='0.0.1',
    description='Time Series Forecasting for Energy Demand Spike Prediction (SPIKE)',
    author='Thore Buergel',
    author_email='tbuergel@pm.me',
    url='https://github.com/thbuerg/spike',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

