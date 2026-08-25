# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LoRA gradient flow through the Nemotron-H Mamba2 mixer's kernel paths.

The Triton kernels are stubbed with differentiable CPU stand-ins that consume
their weight arguments exactly as the real kernels do.
"""

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers.models.nemotron_h import modeling_nemotron_h
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHForCausalLM,
    NemotronHMamba2Mixer,
)

from agilerl.architectures import install_family_zero3_patches
from agilerl.architectures.nemotron_h.mamba import (
    FUSED_PATH_PATCHED_FLAG,
    STREAM_PATCHED_FLAG,
)

CHECKPOINT_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


KERNEL_GLOBALS = ("mamba_split_conv1d_scan_combined", "mamba_chunk_scan_combined")


@pytest.fixture
def pristine_mixer_class():
    """Restore the mixer class and kernel globals (absent until a mixer builds)."""
    saved_init = NemotronHMamba2Mixer.__init__
    saved_forward = NemotronHMamba2Mixer.forward
    saved_kernels = {
        name: getattr(modeling_nemotron_h, name)
        for name in KERNEL_GLOBALS
        if hasattr(modeling_nemotron_h, name)
    }
    yield
    NemotronHMamba2Mixer.__init__ = saved_init
    NemotronHMamba2Mixer.forward = saved_forward
    for name in KERNEL_GLOBALS:
        if name in saved_kernels:
            setattr(modeling_nemotron_h, name, saved_kernels[name])
        elif hasattr(modeling_nemotron_h, name):
            delattr(modeling_nemotron_h, name)
    for flag in (FUSED_PATH_PATCHED_FLAG, STREAM_PATCHED_FLAG):
        if flag in vars(NemotronHMamba2Mixer):
            delattr(NemotronHMamba2Mixer, flag)


def _tiny_nemotron_h():
    config = NemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        layers_block_type=["mamba", "attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=32,
        use_mamba_kernels=False,
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=1,
        conv_kernel=2,
        expand=2,
        chunk_size=8,
        max_position_embeddings=64,
        use_cache=False,
    )
    torch.manual_seed(0)
    return NemotronHForCausalLM(config).float()


def _fake_split_scan(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D=None,
    rmsnorm_weight=None,
    outproj_weight=None,
    outproj_bias=None,
    **kwargs,
):
    hidden = zxbcdt[..., : outproj_weight.shape[1]]
    return F.linear(hidden, outproj_weight, outproj_bias), None


def _fake_chunk_scan(hidden_states, dt, A, B, C, **kwargs):
    return hidden_states, None


def _force_cuda_kernels_path():
    """Route the mixer through ``cuda_kernels_forward`` with CPU kernel stubs.

    Mixer ``__init__`` resets the kernel globals, so call after model build.
    """

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)

    NemotronHMamba2Mixer.forward = forward
    modeling_nemotron_h.mamba_split_conv1d_scan_combined = _fake_split_scan
    modeling_nemotron_h.mamba_chunk_scan_combined = _fake_chunk_scan


def _lora_grad_sums(model, needle):
    return {
        name: None if param.grad is None else float(param.grad.abs().sum())
        for name, param in model.named_parameters()
        if needle in name and ("lora_A" in name or "lora_B" in name)
    }


def _one_backward(base):
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["in_proj", "out_proj"],
        init_lora_weights=False,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config, adapter_name="actor")
    model.train()
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (2, 8))
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    return model


def test_fused_kernel_path_starves_out_proj_lora_of_gradients(pristine_mixer_class):
    base = _tiny_nemotron_h()
    mixer = base.model.layers[0].mixer
    assert mixer.use_mem_eff_path is True
    _force_cuda_kernels_path()

    model = _one_backward(base)

    in_proj = _lora_grad_sums(model, "in_proj")
    out_proj = _lora_grad_sums(model, "out_proj")
    assert len(in_proj) == 2
    assert len(out_proj) == 2
    assert all(value is not None and value > 0 for value in in_proj.values())
    assert all(value is None for value in out_proj.values())


def test_family_patches_after_build_restore_out_proj_lora_gradients(
    pristine_mixer_class,
):
    base = _tiny_nemotron_h()
    patched = install_family_zero3_patches(CHECKPOINT_ID, model=base)
    assert patched == frozenset({"nemotron_h"})
    assert base.model.layers[0].mixer.use_mem_eff_path is False
    _force_cuda_kernels_path()

    model = _one_backward(base)

    grads = _lora_grad_sums(model, "in_proj") | _lora_grad_sums(model, "out_proj")
    assert len(grads) == 4
    assert all(value is not None and value > 0 for value in grads.values())
