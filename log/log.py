import pkg_resources
import logging
import functools
import time
import cProfile, pstats, io


_logger = None
_streamHandler = logging.StreamHandler()


def logger():
    return _logger


def log_args(func):
    @functools.wraps(func)
    def log_args_wrapper(*args, **kwargs):
        args_str = ', '.join(f'{arg}' for arg in args)
        kwargs_str = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
        params_str = ', '.join([s for s in (args_str, kwargs_str) if s])
        logger().info(f'{func.__module__}.{func.__name__}({params_str})')
        return func(*args, **kwargs)
    return log_args_wrapper


def log_exectime(func):
    @functools.wraps(func)
    def log_exectime_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger().info(f'{func.__module__}.{func.__name__} run in {end - start:.3e} sec')
        return result
    return log_exectime_wrapper


def log_profiler_info(sortby='cumulative'):
    def _log_profiler_info(func):
        @functools.wraps(func)
        def log_profiler_info_wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            result = prof.runcall(func, *args, **kwargs)
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
            ps.print_stats()
            logger().info(f'{func.__module__}.{func.__name__} profiler info: {s.getvalue()}')
            return result
        return log_profiler_info_wrapper
    return _log_profiler_info


def start_logging(log_filename=None, logging_level=logging.WARNING):
    global _logger
    _logger = logging.getLogger(__name__)

    current_logging_level = logger().getEffectiveLevel()
    if not current_logging_level or current_logging_level > logging_level:
        logger().setLevel(logging_level)

    if not log_filename:
        handler = _streamHandler
    else:
        logger().removeHandler(_streamHandler)
        handler = logging.FileHandler(str(log_filename))

    handler.setLevel(logging_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - in %(pathname)s:%(funcName)s (line %(lineno)d): %(message)s')
    handler.setFormatter(formatter)
    logger().addHandler(handler)


logfile = pkg_resources.resource_filename('log', 'log.txt')
start_logging(log_filename=logfile, logging_level=logging.INFO)
