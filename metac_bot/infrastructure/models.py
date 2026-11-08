"""OpenCode model reference validation and serialization."""


def model_reference(model: str | None) -> dict[str, str] | None:
    """Validate a provider/model string and convert it to the V2 API shape."""
    if model is None:
        return None
    provider, separator, name = model.partition("/")
    identifier, variant_separator, variant = name.partition("#")
    if not separator or not provider or not identifier:
        raise ValueError(
            "Model must be provider/model, optionally followed by #variant"
        )
    if variant_separator and not variant:
        raise ValueError("Model variant must not be empty")
    result = {"providerID": provider, "id": identifier}
    if variant_separator:
        result["variant"] = variant
    return result
