"""Expected application failures, surfaced with context by the CLI."""


class BenkeError(Exception):
    """Base class for domain failures."""


class DataError(BenkeError):
    """Missing, ambiguous or invalid input data."""


class ModelError(BenkeError):
    """A transformation or model cannot be fitted reliably."""
