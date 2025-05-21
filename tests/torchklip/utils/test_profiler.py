# torchklip/tests/utils/test_profiler.py
import time
import unittest
from unittest import mock

import pytest
import torch
import psutil

from torchklip.utils.profiler import PerformanceMonitor


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for the PerformanceMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use CPU device by default
        self.cpu_device = torch.device("cpu")
        self.cpu_monitor = PerformanceMonitor(device=self.cpu_device)

        # Check if CUDA is available for GPU tests
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            self.gpu_device = torch.device("cuda:0")
            self.gpu_monitor = PerformanceMonitor(device=self.gpu_device)

    def test_timer_functionality(self):
        """Test that timers correctly measure elapsed time."""
        monitor = self.cpu_monitor

        # Start a timer
        monitor.start_timer("test_operation")

        # Sleep for a known duration
        sleep_duration = 0.1
        time.sleep(sleep_duration)

        # Stop the timer and check elapsed time
        elapsed = monitor.stop_timer("test_operation")

        # Allow for some timing imprecision
        self.assertGreaterEqual(elapsed, sleep_duration * 0.9)
        self.assertLessEqual(elapsed, sleep_duration * 1.5)

        # Check that the metric was saved
        self.assertIn("test_operation_time", monitor.metrics)
        self.assertEqual(monitor.metrics["test_operation_time"], elapsed)

    def test_multiple_timers(self):
        """Test that multiple timers can run concurrently."""
        monitor = self.cpu_monitor

        # Start two timers
        monitor.start_timer("operation1")
        time.sleep(0.05)
        monitor.start_timer("operation2")
        time.sleep(0.05)

        # Stop in reverse order
        elapsed2 = monitor.stop_timer("operation2")
        time.sleep(0.05)
        elapsed1 = monitor.stop_timer("operation1")

        # Check timings
        self.assertGreaterEqual(elapsed2, 0.05 * 0.9)
        self.assertLessEqual(elapsed2, 0.05 * 1.5)

        self.assertGreaterEqual(elapsed1, 0.15 * 0.9)
        # More generous bound due to cumulative timing
        self.assertLessEqual(elapsed1, 0.15 * 2)

    def test_stop_nonexistent_timer(self):
        """Test stopping a timer that wasn't started."""
        monitor = self.cpu_monitor

        # Try to stop a timer that doesn't exist
        elapsed = monitor.stop_timer("nonexistent")

        # Should return 0 and not crash
        self.assertEqual(elapsed, 0.0)
        self.assertNotIn("nonexistent_time", monitor.metrics)

    def test_disabled_monitor(self):
        """Test that disabled monitors don't collect metrics."""
        disabled_monitor = PerformanceMonitor(
            device=self.cpu_device, enabled=False)

        # Start and stop a timer
        disabled_monitor.start_timer("test")
        time.sleep(0.1)
        elapsed = disabled_monitor.stop_timer("test")

        # No metrics should be collected
        self.assertEqual(elapsed, 0.0)
        self.assertEqual(disabled_monitor.metrics, {})

    @mock.patch('psutil.Process')
    @mock.patch('psutil.virtual_memory')
    def test_cpu_memory_metrics(self, mock_vm, mock_process):
        """Test collecting CPU memory metrics."""
        # Set up mocks
        mock_process_instance = mock.MagicMock()
        mock_process.return_value = mock_process_instance

        mock_mem_info = mock.MagicMock()
        mock_mem_info.rss = 1024 * 1024 * 100  # 100 MB
        mock_mem_info.vms = 1024 * 1024 * 500  # 500 MB
        mock_process_instance.memory_info.return_value = mock_mem_info

        mock_vm_info = mock.MagicMock()
        mock_vm_info.total = 1024 * 1024 * 1024 * 8  # 8 GB
        mock_vm_info.available = 1024 * 1024 * 1024 * 4  # 4 GB
        mock_vm_info.percent = 50.0
        mock_vm.return_value = mock_vm_info

        # Collect memory metrics
        monitor = self.cpu_monitor
        memory_metrics = monitor.collect_memory_metrics()

        # Check metrics
        self.assertEqual(memory_metrics["cpu_rss"], 1024 * 1024 * 100)
        self.assertEqual(memory_metrics["cpu_vms"], 1024 * 1024 * 500)
        self.assertEqual(
            memory_metrics["system_total_memory"], 1024 * 1024 * 1024 * 8)
        self.assertEqual(
            memory_metrics["system_available_memory"], 1024 * 1024 * 1024 * 4)
        self.assertEqual(memory_metrics["system_percent_used"], 50.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_metrics(self):
        """Test collecting GPU memory metrics (only runs if CUDA is available)."""
        if not self.has_cuda:
            self.skipTest("CUDA not available")

        # Collect memory metrics
        monitor = self.gpu_monitor
        memory_metrics = monitor.collect_memory_metrics()

        # Check that GPU metrics were collected
        self.assertIn("total_gpu_memory", memory_metrics)
        self.assertIn("allocated_gpu_memory", memory_metrics)
        self.assertIn("reserved_gpu_memory", memory_metrics)
        self.assertIn("available_gpu_memory", memory_metrics)

        # Check values are reasonable
        self.assertGreater(memory_metrics["total_gpu_memory"], 0)
        self.assertGreaterEqual(memory_metrics["allocated_gpu_memory"], 0)
        self.assertGreaterEqual(memory_metrics["reserved_gpu_memory"], 0)

        # The available memory should be total minus allocated
        self.assertEqual(
            memory_metrics["available_gpu_memory"],
            memory_metrics["total_gpu_memory"] -
            memory_metrics["allocated_gpu_memory"]
        )

    def test_reset(self):
        """Test resetting metrics and timers."""
        monitor = self.cpu_monitor

        # Add some metrics and timers
        monitor.start_timer("test")
        time.sleep(0.1)
        monitor.stop_timer("test")
        monitor.collect_memory_metrics()

        # Verify we have data
        self.assertGreater(len(monitor.metrics), 0)

        # Reset and verify everything is cleared
        monitor.reset()
        self.assertEqual(monitor.metrics, {})
        self.assertEqual(monitor.timers, {})

    @mock.patch('torchklip.utils.profiler.logger')
    def test_print_metrics(self, mock_logger):
        """Test that metrics printing works without errors."""
        monitor = self.cpu_monitor

        # Add some metrics
        monitor.metrics = {
            "operation1_time": 1.5,
            "operation2_time": 0.5,
            "total_time": 2.0,
            "cpu_rss": 1024 * 1024 * 100,
            "system_total_memory": 1024 * 1024 * 1024 * 8,
            "system_available_memory": 1024 * 1024 * 1024 * 4,
            "system_percent_used": 50.0
        }

        # Call print_metrics and ensure it doesn't error
        monitor.print_metrics()

        # Verify logger was called
        self.assertTrue(mock_logger.info.called)

        # Check for empty metrics
        monitor.reset()
        monitor.print_metrics()
        mock_logger.warning.assert_called_with("No metrics collected")

    def test_str_representation(self):
        """Test string representation of the monitor."""
        monitor = self.cpu_monitor

        # Empty metrics
        self.assertEqual(str(monitor), "No metrics collected")

        # Add some metrics
        monitor.metrics = {
            "operation_time": 1.5,
            "cpu_rss": 1024 * 1024 * 100
        }

        # Check string format
        str_repr = str(monitor)
        self.assertIn("Performance Metrics:", str_repr)
        self.assertIn("Time Metrics:", str_repr)
        self.assertIn("operation_time", str_repr)
        self.assertIn("Memory Metrics:", str_repr)
        self.assertIn("cpu_rss", str_repr)


if __name__ == "__main__":
    unittest.main()
