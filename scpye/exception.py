class Error(Exception):
    """Base-class for all exceptions raised by scpye."""


class ImageNotFoundError(Error):
    """Image not found."""
