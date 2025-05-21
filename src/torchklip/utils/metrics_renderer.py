# torchklip/utils/metrics_renderer.py
import logging
from typing import Optional

# local modules
from .logging_utils import get_logger

# Get a logger specific to this module
logger = get_logger(__name__.split('.')[-1])


def bytes_to_human(num_bytes: int) -> str:
    """Convert a number of bytes into a human-friendly string."""
    # Use 1024 as the conversion factor.
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GB"
    elif num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MB"
    elif num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.2f} KB"
    else:
        return f"{num_bytes} B"


def format_time(duration: float) -> str:
    """
    Convert a duration (in seconds) into a human-friendly format
    using ms, seconds, minutes, or hours as needed.
    """
    if duration < 1e-3:
        return f"{duration * 1e6:.2f} µs"
    elif duration < 1:
        return f"{duration * 1e3:.2f} ms"
    elif duration < 60:
        return f"{duration:.2f} sec"
    elif duration < 3600:
        return f"{duration / 60:.2f} min"
    else:
        return f"{duration / 3600:.2f} hours"


def render_performance_metrics(metrics: dict, logger: Optional[logging.Logger] = None) -> None:
    """
    Render performance metrics in a human-friendly format.

    If a logger is provided, metrics will be emitted at INFO level;
    otherwise they will be printed to stdout.

    Args:
        metrics: Dictionary of metric names to values.
        logger: Optional logger to use for output.
    """
    out = logger.info if logger else print

    if not metrics:
        # if we have a logger, emit warning; otherwise just print
        if logger:
            logger.warning("No metrics collected")
        else:
            print("No metrics collected")
        return

    out("====================================")
    out(" Performance Metrics")
    out("====================================")

    # Memory metrics
    if any(k for k in metrics if 'memory' in k or 'cpu_' in k):
        out("-- Memory Usage --")

        # System memory
        if "system_total_memory" in metrics:
            out(
                f"System Total Memory: {bytes_to_human(metrics['system_total_memory'])}")
        if "system_available_memory" in metrics:
            out(
                f"System Available Memory: {bytes_to_human(metrics['system_available_memory'])}")
        if "system_percent_used" in metrics:
            out(f"System Memory Used: {metrics['system_percent_used']:.1f}%")

        # Process memory
        if "cpu_rss" in metrics:
            out(f"Process RSS Memory: {bytes_to_human(metrics['cpu_rss'])}")
        if "cpu_vms" in metrics:
            out(
                f"Process Virtual Memory: {bytes_to_human(metrics['cpu_vms'])}")

        # GPU memory
        if "total_gpu_memory" in metrics:
            out(
                f"\nGPU Total Memory: {bytes_to_human(metrics['total_gpu_memory'])}")
        if "allocated_gpu_memory" in metrics:
            out(
                f"GPU Allocated Memory: {bytes_to_human(metrics['allocated_gpu_memory'])}")
        if "reserved_gpu_memory" in metrics:
            out(
                f"GPU Reserved Memory: {bytes_to_human(metrics['reserved_gpu_memory'])}")
        if "available_gpu_memory" in metrics:
            out(
                f"GPU Available Memory: {bytes_to_human(metrics['available_gpu_memory'])}")

    out("------------------------------------")

    # Time metrics
    time_metrics = {k: v for k, v in metrics.items() if k.endswith('_time')}
    if time_metrics:
        out("-- Execution Times --")

        total = time_metrics.get("total_time")
        if total is not None:
            out(f"Total Execution Time: {format_time(total)}")

        # Show other metrics, sorted by duration (descending)
        for name, value in sorted(
            ((k, v) for k, v in time_metrics.items() if k != "total_time"),
            key=lambda x: x[1], reverse=True
        ):
            pretty = name.replace('_time', '').replace('_', ' ').title()

            # Show percentage of total time if available
            if total:
                pct = (value / total) * 100
                out(f"- {pretty}: {format_time(value)} ({pct:.1f}% of total)")
            else:
                out(f"- {pretty}: {format_time(value)}")

    out("====================================")


__all__ = ["bytes_to_human", "format_time", "render_performance_metrics"]
