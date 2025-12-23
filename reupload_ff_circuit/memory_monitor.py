"""
Memory monitoring utilities for tracking memory usage and detecting leaks.

This module provides tools to monitor memory consumption during quantum circuit
training and detect potential memory leaks.
"""

import psutil
import os
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
import gc

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage and detect potential leaks."""

    def __init__(self, threshold_mb: float = 100.0):
        """
        Initialize memory monitor.

        Args:
            threshold_mb: Memory increase threshold to warn about (in MB)
        """
        self.process = psutil.Process(os.getpid())
        self.threshold_mb = threshold_mb
        self.initial_memory = None
        self.peak_memory = 0
        self.measurements = []
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def start(self):
        """Start monitoring."""
        gc.collect()  # Force garbage collection
        self.initial_memory = self.get_memory_mb()
        self.peak_memory = self.initial_memory
        self.measurements = [(time.time(), self.initial_memory)]
        logger.info(f"Memory monitoring started: {self.initial_memory:.2f} MB")

    def checkpoint(self, label: str = ""):
        """Record a memory checkpoint."""
        current = self.get_memory_mb()
        self.peak_memory = max(self.peak_memory, current)
        self.measurements.append((time.time(), current))

        if self.initial_memory:
            delta = current - self.initial_memory
            if abs(delta) > self.threshold_mb:
                logger.warning(f"{label} Memory change: {delta:+.2f} MB (Current: {current:.2f} MB)")
            else:
                logger.debug(f"{label} Memory: {current:.2f} MB ({delta:+.2f} MB)")

    def stop(self) -> dict:
        """
        Stop monitoring and return statistics.

        Returns:
            Dictionary with memory statistics
        """
        gc.collect()
        final_memory = self.get_memory_mb()

        stats = {
            'initial_mb': self.initial_memory,
            'final_mb': final_memory,
            'peak_mb': self.peak_memory,
            'delta_mb': final_memory - self.initial_memory if self.initial_memory else 0,
            'peak_increase_mb': self.peak_memory - self.initial_memory if self.initial_memory else 0,
            'measurements': len(self.measurements)
        }

        logger.info(f"Memory monitoring stopped:")
        logger.info(f"  Initial: {stats['initial_mb']:.2f} MB")
        logger.info(f"  Final: {stats['final_mb']:.2f} MB")
        logger.info(f"  Peak: {stats['peak_mb']:.2f} MB")
        logger.info(f"  Delta: {stats['delta_mb']:+.2f} MB")
        logger.info(f"  Peak increase: {stats['peak_increase_mb']:+.2f} MB")

        # Check for potential leak
        if stats['delta_mb'] > self.threshold_mb:
            logger.warning(f"⚠ Potential memory leak detected! Memory increased by {stats['delta_mb']:.2f} MB")
            logger.warning("  Consider using jqc_nq_chunked() for large datasets")

        return stats

    def force_cleanup(self):
        """Force garbage collection and JAX cache cleanup."""
        import gc
        gc.collect()

        try:
            import jax
            jax.clear_caches()
            logger.info("JAX caches cleared")
        except:
            pass

        logger.info("Garbage collection forced")


def monitor_memory(threshold_mb: float = 100.0):
    """
    Decorator to monitor memory usage of a function.

    Args:
        threshold_mb: Memory increase threshold to warn about

    Example:
        @monitor_memory(threshold_mb=50.0)
        def train_model(params, data):
            # training code
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            monitor = MemoryMonitor(threshold_mb=threshold_mb)
            monitor.start()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.stop()
                monitor.force_cleanup()

        return wrapper
    return decorator


class BatchMemoryMonitor:
    """Monitor memory during batch processing."""

    def __init__(self, num_batches: int):
        """
        Initialize batch monitor.

        Args:
            num_batches: Total number of batches to process
        """
        self.num_batches = num_batches
        self.monitor = MemoryMonitor()
        self.batch_memories = []

    def start(self):
        """Start batch monitoring."""
        self.monitor.start()

    def batch_complete(self, batch_idx: int):
        """Record batch completion."""
        current_mb = self.monitor.get_memory_mb()
        self.batch_memories.append(current_mb)

        if len(self.batch_memories) > 1:
            delta = current_mb - self.batch_memories[-2]
            logger.debug(f"Batch {batch_idx+1}/{self.num_batches}: {current_mb:.2f} MB ({delta:+.2f} MB)")

    def check_leak(self) -> bool:
        """
        Check if memory is consistently increasing (potential leak).

        Returns:
            True if potential leak detected
        """
        if len(self.batch_memories) < 5:
            return False

        # Check last 5 batches for consistent increase
        recent = self.batch_memories[-5:]
        increases = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])

        if increases >= 4:  # 4 out of 5 batches increased
            logger.warning(f"⚠ Memory consistently increasing over last {len(recent)} batches")
            logger.warning("  Recommendation: Use jqc_nq_chunked() or reduce batch size")
            return True

        return False

    def finish(self) -> dict:
        """Finish monitoring and return stats."""
        stats = self.monitor.stop()

        if self.batch_memories:
            stats['avg_batch_mb'] = sum(self.batch_memories) / len(self.batch_memories)
            stats['max_batch_mb'] = max(self.batch_memories)
            stats['min_batch_mb'] = min(self.batch_memories)

        return stats


def check_memory_available(required_mb: float = 1000.0) -> bool:
    """
    Check if sufficient memory is available.

    Args:
        required_mb: Required memory in MB

    Returns:
        True if sufficient memory available
    """
    available = psutil.virtual_memory().available / 1024 / 1024

    if available < required_mb:
        logger.warning(f"Low memory: {available:.2f} MB available, {required_mb:.2f} MB required")
        return False

    logger.info(f"Memory check: {available:.2f} MB available")
    return True


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Example 1: Using context manager style
    monitor = MemoryMonitor(threshold_mb=1000.0)
    monitor.start()

    # Simulate some work
    import numpy as np
    data = [np.random.rand(1000, 1000) for _ in range(10)]
    monitor.checkpoint("After data generation")

    # More work
    processed = [d @ d.T for d in data]
    monitor.checkpoint("After processing")

    # Cleanup
    del data, processed
    stats = monitor.stop()

    print(f"\nFinal stats: {stats}")

    # Example 2: Using decorator
    @monitor_memory(threshold_mb=100.0)
    def process_large_data():
        import numpy as np
        return [np.random.rand(2000, 2000) for _ in range(5)]

    result = process_large_data()
