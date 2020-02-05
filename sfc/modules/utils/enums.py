"""Define general DataClasses which are used for timeseries generation and processing for convenience"""
from enum import Enum


class Interval(Enum):
    """Define the possible intervals for time series generation"""
    D = 'd'
    H = 'h'
