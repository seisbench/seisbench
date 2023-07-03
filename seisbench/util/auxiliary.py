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
