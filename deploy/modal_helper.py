"""The helper model (Qwen3.5-9B) on Modal, served by vLLM as an OpenAI-compatible API.

    modal deploy deploy/modal_helper.py          # settings from MODAL_HELPER_* (deploy/helper_config.py)
    modal app stop puddles-helper                # stop it, and the GPU bill

The endpoint is https://<workspace>--puddles-helper-serve.modal.run/v1 . Every request needs
`Authorization: Bearer <VLLM_API_KEY>`, from the Modal Secret `puddles-helper-key`. Clients in this
repo set HELPER_URL, HELPER_MODEL, HELPER_API_KEY and HELPER_EXTRA_BODY; see deploy/README.md.
"""
import os
import subprocess

import modal

from deploy.helper_config import (
    APP_NAME,
    CONTAINER_ENV,
    HF_CACHE_VOLUME,
    PORT,
    SECRET_NAME,
    VLLM_CACHE_VOLUME,
    VLLM_VERSION,
    HelperSettings,
    vllm_command,
)

settings = HelperSettings.from_env(os.environ)  # read where `modal deploy` runs

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .env({**settings.as_env(), **CONTAINER_ENV})  # the container serves what was deployed
    .add_local_python_source("deploy")
)

hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
vllm_cache = modal.Volume.from_name(VLLM_CACHE_VOLUME, create_if_missing=True)

app = modal.App(APP_NAME)

function_options = dict(
    image=image,
    gpu=settings.gpu,
    min_containers=settings.min_containers,
    scaledown_window=settings.scaledown_s,
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
if settings.region:
    function_options["region"] = settings.region


@app.function(**function_options)
@modal.concurrent(max_inputs=settings.concurrency)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    if not os.environ.get("VLLM_API_KEY"):
        # Never serve without auth: the URL is public.
        raise RuntimeError(f"VLLM_API_KEY is not set; create the Modal Secret {SECRET_NAME!r} first")
    subprocess.Popen(vllm_command(HelperSettings.from_env(os.environ)))
