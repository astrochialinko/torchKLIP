# torchklip/tests/utils/test_metrics_renderer.py
import io
import sys
import unittest
from unittest.mock import patch, MagicMock

import pytest

from torchklip.utils.metrics_renderer import (
    bytes_to_human,
    format_time,
    render_performance_metrics,
)


class TestBytesToHuman(unittest.TestCase):
    """Test the bytes_to_human function."""

    def test_bytes(self):
        """Test conversion of bytes."""
        self.assertEqual(bytes_to_human(42), "42 B")
        self.assertEqual(bytes_to_human(0), "0 B")
        self.assertEqual(bytes_to_human(1023), "1023 B")

    def test_kilobytes(self):
        """Test conversion of kilobytes."""
        self.assertEqual(bytes_to_human(1024), "1.00 KB")
        self.assertEqual(bytes_to_human(1536), "1.50 KB")
        self.assertEqual(bytes_to_human(10240), "10.00 KB")

    def test_megabytes(self):
        """Test conversion of megabytes."""
        self.assertEqual(bytes_to_human(1048576), "1.00 MB")
        self.assertEqual(bytes_to_human(5242880), "5.00 MB")
        self.assertEqual(bytes_to_human(1572864), "1.50 MB")

    def test_gigabytes(self):
        """Test conversion of gigabytes."""
        self.assertEqual(bytes_to_human(1073741824), "1.00 GB")
        self.assertEqual(bytes_to_human(1610612736), "1.50 GB")
        self.assertEqual(bytes_to_human(10737418240), "10.00 GB")


class TestFormatTime(unittest.TestCase):
    """Test the format_time function."""

    def test_microseconds(self):
        """Test formatting of microsecond durations."""
        self.assertEqual(format_time(0.0000005), "0.50 µs")
        self.assertEqual(format_time(0.000001), "1.00 µs")
        self.assertEqual(format_time(0.000999), "999.00 µs")

    def test_milliseconds(self):
        """Test formatting of millisecond durations."""
        self.assertEqual(format_time(0.001), "1.00 ms")
        self.assertEqual(format_time(0.5), "500.00 ms")
        self.assertEqual(format_time(0.999), "999.00 ms")

    def test_seconds(self):
        """Test formatting of second durations."""
        self.assertEqual(format_time(1.0), "1.00 sec")
        self.assertEqual(format_time(30.5), "30.50 sec")
        self.assertEqual(format_time(59.999), "60.00 sec")

    def test_minutes(self):
        """Test formatting of minute durations."""
        self.assertEqual(format_time(60.0), "1.00 min")
        self.assertEqual(format_time(90.0), "1.50 min")
        self.assertEqual(format_time(3599.999), "60.00 min")

    def test_hours(self):
        """Test formatting of hour durations."""
        self.assertEqual(format_time(3600.0), "1.00 hours")
        self.assertEqual(format_time(5400.0), "1.50 hours")
        self.assertEqual(format_time(7200.0), "2.00 hours")


class TestRenderPerformanceMetrics(unittest.TestCase):
    """Test the render_performance_metrics function."""

    def setUp(self):
        """Set up the test case."""
        self.sample_metrics = {
            # Memory metrics
            "system_total_memory": 16 * (1 << 30),  # 16 GB
            "system_available_memory": 8 * (1 << 30),  # 8 GB
            "system_percent_used": 50.0,
            "cpu_rss": 500 * (1 << 20),  # 500 MB
            "cpu_vms": 1 * (1 << 30),  # 1 GB
            "total_gpu_memory": 8 * (1 << 30),  # 8 GB
            "allocated_gpu_memory": 2 * (1 << 30),  # 2 GB
            "reserved_gpu_memory": 3 * (1 << 30),  # 3 GB
            "available_gpu_memory": 3 * (1 << 30),  # 3 GB

            # Time metrics
            "total_time": 120.0,  # 2 minutes
            "load_data_time": 10.0,  # 10 seconds
            "preprocess_time": 5.0,  # 5 seconds
            "compute_time": 100.0,  # 100 seconds
            "postprocess_time": 5.0,  # 5 seconds
        }

        # Create a mock logger
        self.mock_logger = MagicMock()
        self.mock_logger.info = MagicMock()
        self.mock_logger.warning = MagicMock()

    def test_render_metrics_to_stdout(self):
        """Test rendering metrics to stdout."""
        # Redirect stdout to capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Call the function with sample metrics
        render_performance_metrics(self.sample_metrics)

        # Get the captured output
        output = captured_output.getvalue()

        # Reset redirect
        sys.stdout = sys.__stdout__

        # Check that expected components are in the output
        self.assertIn("Performance Metrics", output)
        self.assertIn("Memory Usage", output)
        self.assertIn("System Total Memory: 16.00 GB", output)
        self.assertIn("System Available Memory: 8.00 GB", output)
        self.assertIn("System Memory Used: 50.0%", output)
        self.assertIn("Process RSS Memory: 500.00 MB", output)
        self.assertIn("Process Virtual Memory: 1.00 GB", output)
        self.assertIn("GPU Total Memory: 8.00 GB", output)
        self.assertIn("Execution Times", output)
        self.assertIn("Total Execution Time: 2.00 min", output)
        self.assertIn("- Compute: 1.67 min (83.3% of total)", output)

    def test_render_metrics_to_logger(self):
        """Test rendering metrics to a logger."""
        # Call the function with sample metrics and mock logger
        render_performance_metrics(self.sample_metrics, self.mock_logger)

        # Check that logger.info was called with expected messages
        calls = [call[0][0] for call in self.mock_logger.info.call_args_list]

        # Look for " Performance Metrics" with the space at the beginning
        self.assertTrue(any(" Performance Metrics" in call for call in calls),
                        "Performance Metrics header not found in logger calls")

        # Check for a few key metrics in the log calls
        found_memory = False
        found_time = False
        for call in calls:
            if "System Total Memory: 16.00 GB" in call:
                found_memory = True
            if "Total Execution Time: 2.00 min" in call:
                found_time = True

        self.assertTrue(found_memory, "Memory metrics not logged properly")
        self.assertTrue(found_time, "Time metrics not logged properly")

    def test_render_empty_metrics(self):
        """Test rendering empty metrics."""
        # Test with logger
        render_performance_metrics({}, self.mock_logger)
        self.mock_logger.warning.assert_called_once_with(
            "No metrics collected")

        # Test with stdout
        with patch("builtins.print") as mock_print:
            render_performance_metrics({})
            mock_print.assert_called_with("No metrics collected")

    def test_render_partial_metrics(self):
        """Test rendering partial metrics (only some categories)."""
        # Only time metrics
        time_metrics = {
            "total_time": 60.0,
            "process_time": 45.0,
        }

        # Redirect stdout to capture the output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        render_performance_metrics(time_metrics)

        output = captured_output.getvalue()
        sys.stdout = sys.__stdout__

        self.assertIn("Execution Times", output)
        self.assertIn("Total Execution Time: 1.00 min", output)
        self.assertIn("Process: 45.00 sec (75.0% of total)", output)
        self.assertNotIn("Memory Usage", output)

        # Only memory metrics
        memory_metrics = {
            "system_total_memory": 16 * (1 << 30),
            "cpu_rss": 500 * (1 << 20),
        }

        captured_output = io.StringIO()
        sys.stdout = captured_output

        render_performance_metrics(memory_metrics)

        output = captured_output.getvalue()
        sys.stdout = sys.__stdout__

        self.assertIn("Memory Usage", output)
        self.assertIn("System Total Memory: 16.00 GB", output)
        self.assertIn("Process RSS Memory: 500.00 MB", output)
        self.assertNotIn("Execution Times", output)


if __name__ == "__main__":
    unittest.main()
