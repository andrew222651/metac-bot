"""Safe, actionable errors for optional external data providers."""


class ProviderUnavailable(RuntimeError):
    """A provider could not supply data; never include credentials or response bodies."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
