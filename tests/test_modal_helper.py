"""The helper's Modal settings and vLLM command, checked without Modal installed."""
from pathlib import Path

import pytest

from deploy.helper_config import CONTAINER_ENV, ENV, HelperSettings, SettingsError, vllm_command

REPO = Path(__file__).resolve().parents[1]


def test_defaults_serve_qwen_text_only_on_l40s_scaled_to_zero():
    s = HelperSettings.from_env({})
    assert (s.model, s.served_name, s.gpu, s.min_containers, s.region) == (
        "Qwen/Qwen3.5-9B", "qwen3.5-9b", "L40S", 0, "us-east")
    cmd = vllm_command(s)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3.5-9B"]
    assert cmd[cmd.index("--served-model-name") + 1] == "qwen3.5-9b"
    assert "--language-model-only" in cmd
    assert "--enable-lora" not in cmd


def test_the_api_key_is_never_on_the_command_line():
    cmd = vllm_command(HelperSettings.from_env({ENV["lora"]: "1"}))
    assert "--api-key" not in cmd
    assert not any("key" in part.lower() for part in cmd)


def test_lora_flags_only_when_asked():
    cmd = vllm_command(HelperSettings.from_env({ENV["lora"]: "yes", ENV["max_lora_rank"]: "16"}))
    assert cmd[cmd.index("--enable-lora"):] == ["--enable-lora", "--max-lora-rank", "16", "--max-loras", "2"]


def test_settings_round_trip_through_the_container_environment():
    s = HelperSettings.from_env({ENV["gpu"]: "H100", ENV["min_containers"]: "1", ENV["lora"]: "1",
                                 ENV["text_only"]: "no", ENV["region"]: ""})
    assert HelperSettings.from_env(s.as_env()) == s
    assert s.region == "" and s.text_only is False


@pytest.mark.parametrize("env, message", [
    ({ENV["gpu"]: "T4"}, "is not one of"),
    ({ENV["min_containers"]: "5"}, "between 0 and 2"),
    ({ENV["lora"]: "maybe"}, "yes/no"),
    ({ENV["gpu_memory_utilization"]: "0.99"}, "between 0.3 and 0.95"),
    ({ENV["max_model_len"]: "lots"}, "whole number"),
    ({ENV["quantization"]: "int4"}, "must be empty or one of fp8"),
])
def test_bad_settings_are_refused(env, message):
    with pytest.raises(SettingsError, match=message):
        HelperSettings.from_env(env)


def test_the_settings_module_does_not_need_modal():
    source = (REPO / "deploy" / "helper_config.py").read_text()
    assert "import modal" not in source


def test_the_modal_app_refuses_to_serve_without_a_key():
    source = (REPO / "deploy" / "modal_helper.py").read_text()
    assert 'os.environ.get("VLLM_API_KEY")' in source and "raise RuntimeError" in source


def test_fp8_only_when_asked_and_it_survives_the_round_trip():
    assert "--quantization" not in vllm_command(HelperSettings.from_env({}))
    s = HelperSettings.from_env({ENV["gpu"]: "A10G", ENV["quantization"]: "FP8", ENV["max_model_len"]: "4096"})
    cmd = vllm_command(s)
    assert cmd[cmd.index("--quantization") + 1] == "fp8"
    assert HelperSettings.from_env(s.as_env()) == s


def test_the_container_does_not_need_a_cuda_compiler():
    """The first deploy crashed in warm-up: FlashInfer's sampler compiles a kernel and the image has no nvcc."""
    assert CONTAINER_ENV["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "CONTAINER_ENV" in (REPO / "deploy" / "modal_helper.py").read_text()


def test_idle_gpu_time_is_short_by_default():
    assert HelperSettings.from_env({}).scaledown_s == 120
