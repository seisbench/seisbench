import seisbench
import functools


def log_lifecycle(level):
    """
    Logs the invocation and termination of a function to seisbench.logger. Should be used as a decorator.
    """

    def decorator(func):
        @functools.wraps(func)
        def f(*args, **kwargs):
            seisbench.logger.log(msg=f"Starting {func.__name__}", level=level)
            res = func(*args, **kwargs)
            seisbench.logger.log(msg=f"Stopping {func.__name__}", level=level)
            return res

        return f

    return decorator
