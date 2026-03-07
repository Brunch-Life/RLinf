# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone policy server for real-world evaluation.

Loads a trained OpenPI policy and serves it over WebSocket, fully compatible with
the OpenPI client protocol (``openpi_client.websocket_client_policy``).  The eval
script (``eval_realworld.py``) or any OpenPI-compatible client can connect to this
server to query actions.

Supports both ``.safetensors`` (OpenPI native) and ``.pt`` (PyTorch) checkpoint
formats.  When ``--checkpoint-dir`` contains a ``model.safetensors`` file the
safetensors loader is used; when it contains ``*.pt`` files they are loaded via
``torch.load``.  A single ``.pt`` file can also be passed directly via
``--checkpoint-dir``.

Usage
-----
Serve a fine-tuned OpenPI checkpoint (safetensors)::

    python serve_policy.py \\
        --config pi0_custom \\
        --checkpoint-dir /path/to/checkpoint_dir \\
        --port 8000

Serve a PyTorch ``.pt`` checkpoint::

    python serve_policy.py \\
        --config pi0_custom \\
        --checkpoint-dir /path/to/checkpoint.pt \\
        --port 8000

Serve with a default prompt injected when the client doesn't send one::

    python serve_policy.py \\
        --config pi0_custom \\
        --checkpoint-dir /path/to/checkpoint_dir \\
        --default-prompt "pick up the object"
"""

import argparse
import asyncio
import functools
import glob
import http
import logging
import os
import socket
import time
import traceback

import msgpack
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# msgpack-numpy helpers (mirrors openpi_client.msgpack_numpy)
# ---------------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported numpy dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# WebSocket policy server (compatible with OpenPI's WebsocketPolicyServer)
# ---------------------------------------------------------------------------

class WebsocketPolicyServer:
    """Serves a policy over WebSocket using the OpenPI wire format."""

    def __init__(self, policy, *, host="0.0.0.0", port=8000, metadata=None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        import websockets.asyncio.server as ws_server

        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info("Policy server listening on %s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket):
        import websockets

        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error.")
                raise

    @staticmethod
    async def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None


# ---------------------------------------------------------------------------
# Policy loading — supports safetensors, .pt, and OpenPI native
# ---------------------------------------------------------------------------

def _resolve_checkpoint(checkpoint_dir: str) -> tuple[str, str]:
    """Determine checkpoint directory and format.

    Returns:
        (checkpoint_dir, format)  where format is one of:
        "safetensors", "pt_file", "pt_dir"
    """
    if os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith(".pt"):
        return os.path.dirname(checkpoint_dir) or ".", "pt_file"

    if os.path.isdir(checkpoint_dir):
        if os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
            return checkpoint_dir, "safetensors"
        st_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if st_files:
            return checkpoint_dir, "safetensors"
        pt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))
        if pt_files:
            return checkpoint_dir, "pt_dir"

    return checkpoint_dir, "safetensors"


def load_policy_rlinf(
    config_name: str,
    checkpoint_path: str,
    default_prompt: str | None = None,
    num_steps: int = 10,
):
    """Load a policy using RLinf's own config registry and model loading.

    Supports:
    - ``.safetensors`` checkpoint directories (OpenPI native format)
    - ``.pt`` checkpoint files / directories (RLinf training output)

    The ``config_name`` should be one of the registered configs in
    ``rlinf.models.embodiment.openpi.dataconfig``, e.g. ``pi0_custom``.
    """
    import openpi.policies.policy as _policy
    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors.torch
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    train_config = get_openpi_config(config_name, model_path=checkpoint_path)
    checkpoint_dir = download.maybe_download(str(checkpoint_path))
    ckpt_dir, ckpt_fmt = _resolve_checkpoint(checkpoint_dir)

    logger.info("Config: %s | Checkpoint: %s (format: %s)", config_name, ckpt_dir, ckpt_fmt)

    model = pi0_pytorch.PI0Pytorch(config=train_config.model)

    if ckpt_fmt == "safetensors":
        st_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
        if not st_files:
            st_files = [os.path.join(ckpt_dir, "model.safetensors")]
        for sf in st_files:
            logger.info("Loading safetensors: %s", sf)
            safetensors.torch.load_model(model, sf, strict=False)
    elif ckpt_fmt == "pt_file":
        pt_path = checkpoint_path if checkpoint_path.endswith(".pt") else os.path.join(ckpt_dir, checkpoint_path)
        logger.info("Loading .pt checkpoint: %s", pt_path)
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    elif ckpt_fmt == "pt_dir":
        pt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
        for pt_path in pt_files:
            logger.info("Loading .pt checkpoint: %s", pt_path)
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unknown checkpoint format: {ckpt_fmt}")

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    norm_stats = None
    if data_config.asset_id is not None:
        logger.info("Looking for norm_stats (asset_id=%s) ...", data_config.asset_id)
        try:
            norm_stats = _checkpoints.load_norm_stats(ckpt_dir, data_config.asset_id)
            logger.info("✓ norm_stats loaded from checkpoint dir: %s", ckpt_dir)
        except Exception:
            logger.warning(
                "Could not load norm_stats from %s (asset_id=%s). "
                "Trying assets_dirs fallback ...",
                ckpt_dir, data_config.asset_id,
            )
            try:
                norm_stats = _checkpoints.load_norm_stats(
                    train_config.assets_dirs, data_config.asset_id
                )
                logger.info(
                    "✓ norm_stats loaded from assets_dirs: %s",
                    train_config.assets_dirs,
                )
            except Exception:
                logger.error(
                    "✗ norm_stats NOT FOUND anywhere! "
                    "Normalize/Unnormalize will be SKIPPED. "
                    "Model output will be in raw normalized space — actions will be WRONG. "
                    "Make sure the checkpoint directory contains "
                    "assets/<asset_id>/norm_stats.json"
                )
    else:
        logger.warning("data_config.asset_id is None — skipping norm_stats loading.")

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"

    repack_transforms = transforms.Group()
    policy = _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            *(
                [transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)]
                if norm_stats is not None
                else []
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            *(
                [transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm)]
                if norm_stats is not None
                else []
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata=getattr(train_config, "policy_metadata", {}),
        is_pytorch=True,
        pytorch_device=device,
    )
    return policy


def load_policy_openpi_native(
    config_name: str,
    checkpoint_dir: str,
    default_prompt: str | None = None,
):
    """Load a policy using OpenPI's native ``create_trained_policy``."""
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config = _config.get_config(config_name)
    return _policy_config.create_trained_policy(
        config, checkpoint_dir, default_prompt=default_prompt
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve a trained policy over WebSocket (OpenPI-compatible)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pi0_custom",
        help=(
            "OpenPI config name. Use RLinf-registered names like 'pi0_custom', "
            "'pi0_libero', 'pi05_maniskill', etc. (default: pi0_custom)"
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help=(
            "Path to the checkpoint. Can be a directory containing "
            "*.safetensors or *.pt files, or a single .pt file path."
        ),
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt injected when 'prompt' key is absent from input.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of denoising steps for flow-matching inference (default: 10).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000).",
    )
    parser.add_argument(
        "--use-openpi-native",
        action="store_true",
        help=(
            "Use OpenPI's native config registry (openpi.training.config) "
            "instead of RLinf's. For vanilla OpenPI checkpoints."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help=(
            "Number of dummy inferences to run before serving, "
            "triggering AUTOTUNE/JIT so real requests are fast (default: 3)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.use_openpi_native:
        logger.info(
            "Loading with OpenPI native config (config=%s, checkpoint=%s)",
            args.config, args.checkpoint_dir,
        )
        policy = load_policy_openpi_native(
            args.config, args.checkpoint_dir, args.default_prompt
        )
    else:
        logger.info(
            "Loading with RLinf config (config=%s, checkpoint=%s)",
            args.config, args.checkpoint_dir,
        )
        policy = load_policy_rlinf(
            args.config,
            args.checkpoint_dir,
            default_prompt=args.default_prompt,
            num_steps=args.num_steps,
        )

    policy_metadata = getattr(policy, "metadata", {})

    # Warmup: run a few dummy inferences to trigger AUTOTUNE / JIT compilation
    # so that real requests don't pay the first-inference penalty.
    from rlinf.models.embodiment.openpi.policies.franka_policy import make_franka_example

    n_warmup = args.warmup_steps
    if n_warmup > 0:
        logger.info("Running %d warmup inference(s) ...", n_warmup)
        dummy_obs = make_franka_example()
        for i in range(n_warmup):
            t0 = time.monotonic()
            policy.infer(dummy_obs)
            elapsed = (time.monotonic() - t0) * 1000
            logger.info("  warmup %d/%d: %.0f ms", i + 1, n_warmup, elapsed)
        logger.info("Warmup complete. Ready to serve.")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Host: %s, IP: %s", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
