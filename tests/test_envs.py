# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import vllm.envs as envs
from vllm.envs import (
    disable_envs_cache,
    enable_envs_cache,
    env_list_with_choices,
    env_set_with_choices,
    env_with_choices,
    environment_variables,
)


def test_getattr_without_cache(monkeypatch: pytest.MonkeyPatch):
    assert envs.VLLM_HOST_IP == ""
    assert envs.VLLM_PORT is None
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    monkeypatch.setenv("VLLM_PORT", "1234")
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    assert envs.VLLM_PORT == 1234
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")


def test_getattr_with_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    monkeypatch.setenv("VLLM_PORT", "1234")
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    # Enable envs cache and ignore ongoing environment changes
    enable_envs_cache()

    # __getattr__ is decorated with functools.cache
    assert hasattr(envs.__getattr__, "cache_info")
    start_hits = envs.__getattr__.cache_info().hits

    # 2 more hits due to VLLM_HOST_IP and VLLM_PORT accesses
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    assert envs.VLLM_PORT == 1234
    assert envs.__getattr__.cache_info().hits == start_hits + 2

    # All environment variables are cached
    for environment_variable in environment_variables:
        envs.__getattr__(environment_variable)
    assert envs.__getattr__.cache_info().hits == start_hits + 2 + len(
        environment_variables
    )

    # Reset envs.__getattr__ back to none-cached version to
    # avoid affecting other tests
    envs.__getattr__ = envs.__getattr__.__wrapped__


def test_getattr_with_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_HOST_IP", "1.1.1.1")
    # __getattr__ is not decorated with functools.cache
    assert not hasattr(envs.__getattr__, "cache_info")

    # Enable envs cache and ignore ongoing environment changes
    enable_envs_cache()
    assert envs.VLLM_HOST_IP == "1.1.1.1"
    # With cache enabled, the environment variable value is cached and unchanged
    monkeypatch.setenv("VLLM_HOST_IP", "2.2.2.2")
    assert envs.VLLM_HOST_IP == "1.1.1.1"

    disable_envs_cache()
    assert envs.VLLM_HOST_IP == "2.2.2.2"
    # After cache disabled, the environment variable value would be synced
    # with os.environ
    monkeypatch.setenv("VLLM_HOST_IP", "3.3.3.3")
    assert envs.VLLM_HOST_IP == "3.3.3.3"


def test_is_envs_cache_enabled() -> None:
    assert not envs._is_envs_cache_enabled()
    enable_envs_cache()
    assert envs._is_envs_cache_enabled()

    # Only wrap one-layer of cache, so we only need to
    # call disable once to reset.
    enable_envs_cache()
    enable_envs_cache()
    enable_envs_cache()
    disable_envs_cache()
    assert not envs._is_envs_cache_enabled()

    disable_envs_cache()
    assert not envs._is_envs_cache_enabled()


def test_lk_moe_gpu_resident_layer_plan_parse(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LVLLM_MOE_QUANT_ON_GPU", "1")
    monkeypatch.setenv(
        "LVLLM_GPU_RESIDENT_MOE_LAYERS",
        "1, 3-5, 7, bad, 9-8, 11",
    )

    assert envs.get_lk_moe_gpu_resident_layer_plan() == frozenset(
        {1, 3, 4, 5, 7, 11}
    )
    assert envs.is_lk_moe_gpu_resident_layer_idx(4)
    assert not envs.is_lk_moe_gpu_resident_layer_idx(6)


def test_lk_moe_gpu_resident_layer_idx_disabled_feature(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LVLLM_MOE_QUANT_ON_GPU", "0")
    monkeypatch.setenv("LVLLM_GPU_RESIDENT_MOE_LAYERS", "")

    # Feature disabled means all layers stay on GPU by default.
    assert envs.is_lk_moe_gpu_resident_layer_idx(123)


def test_validate_lvllm_config_contract_defaults(monkeypatch: pytest.MonkeyPatch):
    disable_envs_cache()
    for name in (
        "LVLLM_MOE_NUMA_ENABLED",
        "LVLLM_ENABLE_NUMA_INTERLEAVE",
        "LVLLM_NUMA_BIND_STRATEGY",
        "LVLLM_NUMACTL_ARGS_OVERRIDE",
        "LVLLM_MOE_QUANT_ON_GPU",
        "LVLLM_HW_AWARE_TUNING",
        "LVLLM_MOE_USE_WEIGHT",
        "LVLLM_GPU_RESIDENT_MOE_LAYERS",
        "LVLLM_GPU_PREFILL_MIN_BATCH_SIZE",
        "LVLLM_GPU_PREFETCH_WINDOW",
    ):
        monkeypatch.delenv(name, raising=False)

    snapshot = envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)
    assert snapshot["LVLLM_MOE_USE_WEIGHT"] == "INT4"
    assert snapshot["LVLLM_GPU_PREFETCH_WINDOW"] == 3
    assert snapshot["LVLLM_GPU_PREFILL_MIN_BATCH_SIZE"] == 0
    assert snapshot["LVLLM_GPU_PREFILL_ENABLED"] is False


def test_validate_lvllm_config_contract_uses_hardware_recommendation(
    monkeypatch: pytest.MonkeyPatch,
):
    disable_envs_cache()
    monkeypatch.setenv("LVLLM_MOE_NUMA_ENABLED", "1")
    monkeypatch.setenv("LVLLM_HW_AWARE_TUNING", "1")
    monkeypatch.delenv("LVLLM_GPU_PREFILL_MIN_BATCH_SIZE", raising=False)
    monkeypatch.delenv("LVLLM_GPU_PREFETCH_WINDOW", raising=False)

    monkeypatch.setattr(envs, "is_lk_moe_feature_enabled", lambda: True)
    monkeypatch.setattr(
        envs,
        "get_lvllm_hardware_recommendations",
        lambda *args, **kwargs: {
            "LK_THREADS": 44,
            "OMP_NUM_THREADS": 44,
            "LVLLM_GPU_PREFILL_MIN_BATCH_SIZE": 2048,
            "LVLLM_GPU_PREFETCH_WINDOW": 1,
            "GPU_COUNT": 2,
            "GPU_MIN_MEMORY_GB": 24,
            "GPU_MIN_COMPUTE_CAPABILITY": 89,
            "CPU_LOGICAL_CORES": 192,
            "CPU_PHYSICAL_CORES": 96,
            "WORKER_PROCESSES": 2,
        },
    )

    snapshot = envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)
    assert snapshot["LVLLM_GPU_PREFILL_MIN_BATCH_SIZE"] == 2048
    assert snapshot["LVLLM_GPU_PREFETCH_WINDOW"] == 1
    assert snapshot["LVLLM_GPU_PREFILL_MIN_BATCH_SIZE_SOURCE"] == "hardware_aware"
    assert snapshot["LVLLM_GPU_PREFETCH_WINDOW_SOURCE"] == "hardware_aware"
    assert snapshot["LK_THREADS_RECOMMENDED"] == 44
    assert snapshot["OMP_NUM_THREADS_RECOMMENDED"] == 44


def test_validate_lvllm_config_contract_prefers_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
):
    disable_envs_cache()
    monkeypatch.setenv("LVLLM_MOE_NUMA_ENABLED", "1")
    monkeypatch.setenv("LVLLM_HW_AWARE_TUNING", "1")
    monkeypatch.setenv("LVLLM_GPU_PREFILL_MIN_BATCH_SIZE", "1024")
    monkeypatch.setenv("LVLLM_GPU_PREFETCH_WINDOW", "2")

    monkeypatch.setattr(envs, "is_lk_moe_feature_enabled", lambda: True)
    monkeypatch.setattr(
        envs,
        "get_lvllm_hardware_recommendations",
        lambda *args, **kwargs: {
            "LK_THREADS": 32,
            "OMP_NUM_THREADS": 32,
            "LVLLM_GPU_PREFILL_MIN_BATCH_SIZE": 2048,
            "LVLLM_GPU_PREFETCH_WINDOW": 1,
            "GPU_COUNT": 1,
            "GPU_MIN_MEMORY_GB": 24,
            "GPU_MIN_COMPUTE_CAPABILITY": 89,
            "CPU_LOGICAL_CORES": 64,
            "CPU_PHYSICAL_CORES": 32,
            "WORKER_PROCESSES": 1,
        },
    )

    snapshot = envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)
    assert snapshot["LVLLM_GPU_PREFILL_MIN_BATCH_SIZE"] == 1024
    assert snapshot["LVLLM_GPU_PREFETCH_WINDOW"] == 2
    assert snapshot["LVLLM_GPU_PREFILL_MIN_BATCH_SIZE_SOURCE"] == "env"
    assert snapshot["LVLLM_GPU_PREFETCH_WINDOW_SOURCE"] == "env"


def test_validate_lvllm_config_contract_rejects_invalid_strategy(
    monkeypatch: pytest.MonkeyPatch,
):
    disable_envs_cache()
    monkeypatch.setenv("LVLLM_MOE_USE_WEIGHT", "INVALID")
    with pytest.raises(ValueError, match="Invalid value 'INVALID' for LVLLM_MOE_USE_WEIGHT"):
        envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)


def test_validate_lvllm_config_contract_rejects_prefetch_conflict(
    monkeypatch: pytest.MonkeyPatch,
):
    disable_envs_cache()
    monkeypatch.setenv("LVLLM_GPU_PREFILL_MIN_BATCH_SIZE", "2048")
    monkeypatch.setenv("LVLLM_GPU_PREFETCH_WINDOW", "0")
    with pytest.raises(
        ValueError,
        match="LVLLM_GPU_PREFETCH_WINDOW must be > 0 when LVLLM_GPU_PREFILL_MIN_BATCH_SIZE > 0",
    ):
        envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)


def test_validate_lvllm_config_contract_requires_moe_enabled_for_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    disable_envs_cache()
    monkeypatch.setenv("LVLLM_MOE_NUMA_ENABLED", "0")
    monkeypatch.setenv("LVLLM_GPU_PREFILL_MIN_BATCH_SIZE", "2048")
    monkeypatch.setenv("LVLLM_GPU_PREFETCH_WINDOW", "1")
    with pytest.raises(
        ValueError,
        match="LVLLM_GPU_PREFILL_MIN_BATCH_SIZE > 0 requires LVLLM_MOE_NUMA_ENABLED=1",
    ):
        envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)


def test_validate_lvllm_config_contract_rejects_invalid_resident_layers(
    monkeypatch: pytest.MonkeyPatch,
):
    disable_envs_cache()
    monkeypatch.setenv("LVLLM_GPU_RESIDENT_MOE_LAYERS", "0-2,bad")
    with pytest.raises(ValueError, match="Invalid LVLLM_GPU_RESIDENT_MOE_LAYERS segments"):
        envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)


def test_validate_lvllm_config_contract_requires_interleave_for_numactl_override(
    monkeypatch: pytest.MonkeyPatch,
):
    disable_envs_cache()
    monkeypatch.setenv("LVLLM_ENABLE_NUMA_INTERLEAVE", "0")
    monkeypatch.setenv("LVLLM_NUMACTL_ARGS_OVERRIDE", "--interleave=all")
    with pytest.raises(
        ValueError,
        match="LVLLM_NUMACTL_ARGS_OVERRIDE requires LVLLM_ENABLE_NUMA_INTERLEAVE=1",
    ):
        envs.validate_lvllm_config_contract(hard_fail=True, log_snapshot=False)


def test_lvllm_readme_env_table_is_synced():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "sync_lvllm_readme_env_table.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lvllm_readme_support_matrix_is_synced():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "sync_lvllm_readme_support_matrix.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class TestEnvWithChoices:
    """Test cases for env_with_choices function."""

    def test_default_value_returned_when_env_not_set(self):
        """Test default is returned when env var is not set."""
        env_func = env_with_choices(
            "NONEXISTENT_ENV", "default", ["option1", "option2"]
        )
        assert env_func() == "default"

    def test_none_default_returned_when_env_not_set(self):
        """Test that None is returned when env not set and default is None."""
        env_func = env_with_choices("NONEXISTENT_ENV", None, ["option1", "option2"])
        assert env_func() is None

    def test_valid_value_returned_case_sensitive(self):
        """Test that valid value is returned in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            assert env_func() == "option1"

    def test_valid_lowercase_value_returned_case_insensitive(self):
        """Test that lowercase value is accepted in case insensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["OPTION1", "OPTION2"], case_sensitive=False
            )
            assert env_func() == "option1"

    def test_valid_uppercase_value_returned_case_insensitive(self):
        """Test that uppercase value is accepted in case insensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == "OPTION1"

    def test_invalid_value_raises_error_case_sensitive(self):
        """Test that invalid value raises ValueError in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()

    def test_case_mismatch_raises_error_case_sensitive(self):
        """Test that case mismatch raises ValueError in case sensitive mode."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(
                ValueError, match="Invalid value 'OPTION1' for TEST_ENV"
            ):
                env_func()

    def test_invalid_value_raises_error_case_insensitive(self):
        """Test that invalid value raises ValueError when case insensitive."""
        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices(
                "TEST_ENV", "default", ["option1", "option2"], case_sensitive=False
            )
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1"}):
            env_func = env_with_choices("TEST_ENV", "default", get_choices)
            assert env_func() == "dynamic1"

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "invalid"}):
            env_func = env_with_choices("TEST_ENV", "default", get_choices)
            with pytest.raises(
                ValueError, match="Invalid value 'invalid' for TEST_ENV"
            ):
                env_func()


class TestEnvListWithChoices:
    """Test cases for env_list_with_choices function."""

    def test_default_list_returned_when_env_not_set(self):
        """Test that default list is returned when env var is not set."""
        env_func = env_list_with_choices(
            "NONEXISTENT_ENV", ["default1", "default2"], ["option1", "option2"]
        )
        assert env_func() == ["default1", "default2"]

    def test_empty_default_list_returned_when_env_not_set(self):
        """Test that empty default list is returned when env not set."""
        env_func = env_list_with_choices("NONEXISTENT_ENV", [], ["option1", "option2"])
        assert env_func() == []

    def test_single_valid_value_parsed_correctly(self):
        """Test that single valid value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1"]

    def test_multiple_valid_values_parsed_correctly(self):
        """Test that multiple valid values are parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_values_with_whitespace_trimmed(self):
        """Test that values with whitespace are trimmed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": " option1 , option2 "}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_empty_values_filtered_out(self):
        """Test that empty values are filtered out."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,,option2,"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option2"]

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ""}):
            env_func = env_list_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == ["default"]

    def test_only_commas_returns_default(self):
        """Test that string with only commas returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ",,,"}):
            env_func = env_list_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == ["default"]

    def test_case_sensitive_validation(self):
        """Test case sensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,OPTION2"}):
            env_func = env_list_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(ValueError, match="Invalid value 'OPTION2' in TEST_ENV"):
                env_func()

    def test_case_insensitive_validation(self):
        """Test case insensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}):
            env_func = env_list_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == ["OPTION1", "option2"]

    def test_invalid_value_in_list_raises_error(self):
        """Test that invalid value in list raises ValueError."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,invalid,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,dynamic2"}):
            env_func = env_list_with_choices("TEST_ENV", [], get_choices)
            assert env_func() == ["dynamic1", "dynamic2"]

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,invalid"}):
            env_func = env_list_with_choices("TEST_ENV", [], get_choices)
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_duplicate_values_preserved(self):
        """Test that duplicate values in the list are preserved."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option1,option2"}):
            env_func = env_list_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == ["option1", "option1", "option2"]


class TestEnvSetWithChoices:
    """Test cases for env_set_with_choices function."""

    def test_default_list_returned_when_env_not_set(self):
        """Test that default list is returned when env var is not set."""
        env_func = env_set_with_choices(
            "NONEXISTENT_ENV", ["default1", "default2"], ["option1", "option2"]
        )
        assert env_func() == {"default1", "default2"}

    def test_empty_default_list_returned_when_env_not_set(self):
        """Test that empty default list is returned when env not set."""
        env_func = env_set_with_choices("NONEXISTENT_ENV", [], ["option1", "option2"])
        assert env_func() == set()

    def test_single_valid_value_parsed_correctly(self):
        """Test that single valid value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1"}

    def test_multiple_valid_values_parsed_correctly(self):
        """Test that multiple valid values are parsed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_values_with_whitespace_trimmed(self):
        """Test that values with whitespace are trimmed correctly."""
        with patch.dict(os.environ, {"TEST_ENV": " option1 , option2 "}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_empty_values_filtered_out(self):
        """Test that empty values are filtered out."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,,option2,"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}

    def test_empty_string_returns_default(self):
        """Test that empty string returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ""}):
            env_func = env_set_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == {"default"}

    def test_only_commas_returns_default(self):
        """Test that string with only commas returns default."""
        with patch.dict(os.environ, {"TEST_ENV": ",,,"}):
            env_func = env_set_with_choices(
                "TEST_ENV", ["default"], ["option1", "option2"]
            )
            assert env_func() == {"default"}

    def test_case_sensitive_validation(self):
        """Test case sensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,OPTION2"}):
            env_func = env_set_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=True
            )
            with pytest.raises(ValueError, match="Invalid value 'OPTION2' in TEST_ENV"):
                env_func()

    def test_case_insensitive_validation(self):
        """Test case insensitive validation."""
        with patch.dict(os.environ, {"TEST_ENV": "OPTION1,option2"}):
            env_func = env_set_with_choices(
                "TEST_ENV", [], ["option1", "option2"], case_sensitive=False
            )
            assert env_func() == {"OPTION1", "option2"}

    def test_invalid_value_in_list_raises_error(self):
        """Test that invalid value in list raises ValueError."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,invalid,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_callable_choices_resolved_correctly(self):
        """Test that callable choices are resolved correctly."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,dynamic2"}):
            env_func = env_set_with_choices("TEST_ENV", [], get_choices)
            assert env_func() == {"dynamic1", "dynamic2"}

    def test_callable_choices_with_invalid_value(self):
        """Test that callable choices raise error for invalid values."""

        def get_choices():
            return ["dynamic1", "dynamic2"]

        with patch.dict(os.environ, {"TEST_ENV": "dynamic1,invalid"}):
            env_func = env_set_with_choices("TEST_ENV", [], get_choices)
            with pytest.raises(ValueError, match="Invalid value 'invalid' in TEST_ENV"):
                env_func()

    def test_duplicate_values_deduped(self):
        """Test that duplicate values in the list are deduped."""
        with patch.dict(os.environ, {"TEST_ENV": "option1,option1,option2"}):
            env_func = env_set_with_choices("TEST_ENV", [], ["option1", "option2"])
            assert env_func() == {"option1", "option2"}


class TestVllmConfigureLogging:
    """Test cases for VLLM_CONFIGURE_LOGGING environment variable."""

    def test_configure_logging_defaults_to_true(self):
        """Test that VLLM_CONFIGURE_LOGGING defaults to True when not set."""
        # Ensure the env var is not set
        with patch.dict(os.environ, {}, clear=False):
            if "VLLM_CONFIGURE_LOGGING" in os.environ:
                del os.environ["VLLM_CONFIGURE_LOGGING"]

            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_zero_string(self):
        """Test that VLLM_CONFIGURE_LOGGING='0' evaluates to False."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "0"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is False
            assert isinstance(result, bool)

    def test_configure_logging_with_one_string(self):
        """Test that VLLM_CONFIGURE_LOGGING='1' evaluates to True."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "1"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            result = envs.VLLM_CONFIGURE_LOGGING
            assert result is True
            assert isinstance(result, bool)

    def test_configure_logging_with_invalid_value_raises_error(self):
        """Test that invalid VLLM_CONFIGURE_LOGGING value raises ValueError."""
        with patch.dict(os.environ, {"VLLM_CONFIGURE_LOGGING": "invalid"}):
            # Clear cache if it exists
            if hasattr(envs.__getattr__, "cache_clear"):
                envs.__getattr__.cache_clear()

            with pytest.raises(ValueError, match="invalid literal for int"):
                _ = envs.VLLM_CONFIGURE_LOGGING
