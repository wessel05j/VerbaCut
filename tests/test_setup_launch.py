from __future__ import annotations

import logging
import sys
import unittest
from unittest.mock import patch

import setup_env
from utils.model_selector import ensure_ollama_running


class SetupStateTests(unittest.TestCase):
    def test_auto_torch_cpu_fallback_skips_future_setup(self) -> None:
        state = {
            "python_executable": sys.executable,
            "requirements_hash": "abc123",
            "torch_request": "auto",
            "torch_target": "cuda",
            "torch_installed_mode": "cpu",
        }

        should_skip = setup_env.should_skip_setup(
            state=state,
            req_hash="abc123",
            requested_torch="auto",
            resolved_torch_target="cuda",
            torch_info={"installed": True, "cuda_available": False},
            force=False,
        )

        self.assertTrue(should_skip)

    def test_explicit_cuda_does_not_accept_cpu_fallback(self) -> None:
        state = {
            "python_executable": sys.executable,
            "requirements_hash": "abc123",
            "torch_request": "cuda",
            "torch_target": "cuda",
            "torch_installed_mode": "cpu",
        }

        should_skip = setup_env.should_skip_setup(
            state=state,
            req_hash="abc123",
            requested_torch="cuda",
            resolved_torch_target="cuda",
            torch_info={"installed": True, "cuda_available": False},
            force=False,
        )

        self.assertFalse(should_skip)


class OllamaStartupTests(unittest.TestCase):
    def _logger(self) -> logging.Logger:
        logger = logging.getLogger(f"ollama-startup-test-{id(self)}")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    @patch("utils.model_selector._start_ollama_process", return_value=True)
    @patch("utils.model_selector._ollama_start_commands", return_value=[("ollama serve", ["ollama", "serve"])])
    @patch("utils.model_selector.ollama_api_reachable", side_effect=[False, True])
    def test_starts_ollama_when_api_is_down(
        self,
        api_reachable,
        start_commands,
        start_process,
    ) -> None:
        self.assertTrue(ensure_ollama_running("http://localhost:11434", self._logger()))
        start_commands.assert_called_once()
        start_process.assert_called_once_with(["ollama", "serve"], self._logger())
        self.assertEqual(api_reachable.call_count, 2)

    @patch("utils.model_selector._ollama_start_commands", return_value=[])
    @patch("utils.model_selector.ollama_api_reachable", return_value=False)
    def test_returns_false_when_no_ollama_executable_is_available(
        self,
        api_reachable,
        start_commands,
    ) -> None:
        self.assertFalse(ensure_ollama_running("http://localhost:11434", self._logger()))
        start_commands.assert_called_once()
        api_reachable.assert_called_once()


if __name__ == "__main__":
    unittest.main()
