import time
# Benchmark save/load times


def benchmark_save_transient_group(num_transients: int=100) -> float:
    """Benchmark how long