# torchklip/utils/profiler.py
import time
from typing import Dict

import psutil
import torch

# local modules
from .logging_utils import get_logger
from .metrics_renderer import bytes_to_human, format_time, render_performance_metrics


# Get a logger specific to this module
logger = get_logger(__name__.split('.')[-1])


class PerformanceMonitor:
    """Utility class for monitoring performance metrics including time and memory usage.

    This can be used to monitor both CPU and GPU operations.
    """

    def __init__(self, device: torch.device, enabled: bool = True):
        """
        Initialize the performance monitor.

        Args:
            device (torch.device): The compute device (CPU or CUDA).
            enabled (bool): Whether to collect metrics or not.
        """
        self.device = device
        self.enabled = enabled
        self.metrics = {}
        self.timers = {}

    def start_timer(self, name: str) -> None:
        """
        Start a named timer.

        Args:
            name (str): Name of the timer.
        """
        if not self.enabled:
            return
        self.timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and record the elapsed time.

        Args:
            name (str): Name of the timer.

        Returns:
            float: Elapsed time in seconds.
        """
        if not self.enabled or name not in self.timers:
            return 0.0

        elapsed = time.perf_counter() - self.timers[name]
        self.metrics[f"{name}_time"] = elapsed
        return elapsed

    def collect_memory_metrics(self) -> Dict[str, int]:
        """
        Collect memory metrics for the current device.

        Returns:
            Dict[str, int]: Dictionary of memory metrics.
        """
        if not self.enabled:
            return {}

        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # Ensure all GPU operations are finished
            total_memory = torch.cuda.get_device_properties(
                self.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            reserved_memory = torch.cuda.memory_reserved(self.device)
            available_memory = total_memory - allocated_memory

            memory_metrics = {
                'total_gpu_memory': total_memory,
                'allocated_gpu_memory': allocated_memory,
                'reserved_gpu_memory': reserved_memory,
                'available_gpu_memory': available_memory
            }
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            vm = psutil.virtual_memory()

            memory_metrics = {
                'cpu_rss': mem_info.rss,          # Resident set size
                'cpu_vms': mem_info.vms,          # Virtual memory size
                'system_total_memory': vm.total,
                'system_available_memory': vm.available,
                'system_percent_used': vm.percent
            }

        self.metrics.update(memory_metrics)
        return memory_metrics

    def get_metrics(self) -> Dict[str, float]:
        """
        Get all collected metrics.

        Returns:
            Dict[str, float]: Dictionary of all metrics.
        """
        return self.metrics

    def reset(self) -> None:
        """Reset all metrics and timers."""
        self.metrics = {}
        self.timers = {}

    def print_metrics(self) -> None:
        """Print all collected metrics in a human-friendly format."""
        render_performance_metrics(self.metrics, logger=logger)

    def __str__(self) -> str:
        """
        String representation of the metrics.

        Returns:
            str: Formatted metrics string.
        """
        if not self.metrics:
            return "No metrics collected"

        result = "Performance Metrics:\n"

        # Format time metrics
        time_metrics = {k: v for k, v in self.metrics.items()
                        if k.endswith('_time')}
        if time_metrics:
            result += "  Time Metrics:\n"
            for name, value in time_metrics.items():
                result += f"    {name}: {format_time(value)}\n"

        # Format memory metrics
        memory_metrics = {
            k: v for k, v in self.metrics.items() if 'memory' in k or 'cpu_' in k}
        if memory_metrics:
            result += "  Memory Metrics:\n"
            for name, value in memory_metrics.items():
                if isinstance(value, (int, float)) and name != "system_percent_used":
                    result += f"    {name}: {bytes_to_human(value)}\n"
                else:
                    result += f"    {name}: {value}\n"

        return result


__all__ = ["PerformanceMonitor"]
