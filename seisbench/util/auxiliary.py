def in_notebook() -> bool:
    """
    Checks whether code is executed within a jupyter notebook
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False  # Not in jupyter notebook
    except ImportError:
        return False  # IPython not installed
    except AttributeError:
        return False  # Not in IPython environment
    return True


class MissingOptionalDependency:
    def __init__(self, name: str, extra: str):
        self._name = name
        self._extra = extra

    def __getattr__(self, attr):
        self._raise()

    def __call__(self, *args, **kwargs):
        self._raise()

    def _raise(self):
        raise ImportError(
            f"Optional dependency '{self._name}' is required. "
            f"Install it with: pip install {self._extra}"
        )
