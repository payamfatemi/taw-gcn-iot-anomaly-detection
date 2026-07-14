class TAWGCNError(Exception):
    """Base project exception."""


class ConfigurationError(TAWGCNError):
    """Raised when configuration is missing or inconsistent."""


class DataIntegrityError(TAWGCNError):
    """Raised when a dataset fails integrity checks."""


class GraphValidationError(TAWGCNError):
    """Raised when a graph violates configured quality constraints."""
