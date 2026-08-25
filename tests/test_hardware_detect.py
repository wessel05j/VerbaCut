from __future__ import annotations

from unittest import TestCase, mock

from utils.hardware_detect import _detect_nvidia_cuda_memory


class CudaMemoryDetectionTests(TestCase):
    def test_nvidia_smi_aggregates_available_memory_across_devices(self) -> None:
        output = "\n".join(
            [
                "0, NVIDIA RTX A, 24576, 4096, 20480",
                "1, NVIDIA RTX B, 16384, 12288, 4096",
            ]
        )

        with mock.patch("utils.hardware_detect._run_command", return_value=output) as run_command:
            snapshot = _detect_nvidia_cuda_memory()

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.device_index, -1)
        self.assertEqual(snapshot.device_count, 2)
        self.assertEqual(snapshot.device_name, "2 CUDA GPUs (NVIDIA RTX A, NVIDIA RTX B)")
        self.assertEqual(snapshot.total_gb, 40.0)
        self.assertEqual(snapshot.free_gb, 16.0)
        self.assertEqual(snapshot.used_gb, 24.0)
        self.assertEqual(snapshot.source, "nvidia-smi")
        self.assertEqual(snapshot.free_fraction, 0.4)
        command = run_command.call_args.args[0]
        self.assertIn("memory.free", command[1])

    def test_nvidia_smi_ignores_malformed_rows(self) -> None:
        output = "invalid\n0, NVIDIA RTX, 8192, unavailable, 100"

        with mock.patch("utils.hardware_detect._run_command", return_value=output):
            snapshot = _detect_nvidia_cuda_memory()

        self.assertIsNone(snapshot)
