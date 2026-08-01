class RuntimeEngineError(RuntimeError):
    """Base error for runtime engine failures."""


class ConfigurationError(RuntimeEngineError):
    """Raised when runtime configuration is invalid."""


class CompatibilityError(RuntimeEngineError):
    """Raised when the selected ComfyUI version is unsupported."""
