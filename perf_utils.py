# perf_utils.py

import time
import logging

# Optional memory usage tracking via psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def measure_time(func):
    """
    Decorator to measure the execution time of a function and log it.
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"[PERF] {func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

def log_memory_usage(label=""):
    """
    Logs current memory usage if psutil is installed.
    """
    if HAS_PSUTIL:
        mem_info = psutil.virtual_memory()
        logging.info(
            f"[MEMORY] {label} usage: {mem_info.percent:.2f}% "
            f"(Available: {mem_info.available // (1024*1024)} MB, "
            f"Total: {mem_info.total // (1024*1024)} MB)"
        )
