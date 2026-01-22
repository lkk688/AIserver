class AppError(Exception):
    """Base class for application errors."""
    pass

class NotFound(AppError):
    """Resource not found."""
    pass

class Conflict(AppError):
    """Resource conflict."""
    pass

class BackendUnavailable(AppError):
    """External service or backend unavailable."""
    pass

class ValidationError(AppError):
    """Domain validation error."""
    pass

class ExtractionError(AppError):
    """Content extraction failed."""
    pass
