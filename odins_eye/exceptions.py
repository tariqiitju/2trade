"""
Custom exceptions for Odin's Eye library
"""


class OdinsEyeError(Exception):
    """Base exception for all Odin's Eye errors"""
    pass


class DataNotFoundError(OdinsEyeError):
    """Raised when requested data is not found"""
    pass


class InvalidFilterError(OdinsEyeError):
    """Raised when an invalid filter is provided"""
    pass


class ConfigurationError(OdinsEyeError):
    """Raised when there's a configuration issue"""
    pass