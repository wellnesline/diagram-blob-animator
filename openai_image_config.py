"""synthetic stub: image generation defaults (constants and a validator only)."""
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_IMAGE_QUALITY = "low"

_ALLOWED_QUALITY = {"low", "medium", "high", "auto"}


def validate_image_defaults(model: str, quality: str) -> None:
    """Raise ValueError when model / quality are empty or quality is not a known tier."""
    if not model:
        raise ValueError("image model must not be empty")
    if quality not in _ALLOWED_QUALITY:
        raise ValueError(f"unsupported image quality: {quality!r}")