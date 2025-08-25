"""
Custom exceptions for the Ramanujan ML framework
"""


class RamanujanError(Exception):
    """Base exception for all Ramanujan errors"""
    pass


class ModelError(RamanujanError):
    """Raised when there's an issue with model operations"""
    pass


class TrainingError(RamanujanError):
    """Raised when training fails or encounters issues"""
    pass


class PredictionError(RamanujanError):
    """Raised when prediction fails"""
    pass


class ConfigurationError(RamanujanError):
    """Raised when there's a configuration issue"""
    pass


class DataError(RamanujanError):
    """Raised when there's an issue with input data"""
    pass