# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Native ``mimo_v2`` loading as :class:`MiMoV2ForConditionalGeneration`: the vision tower and the transformers ``MiMoV2Flash`` language model.

The ``XiaomiMiMo/MiMo-V2.6-Flash`` checkpoints declare ``model_type: mimo_v2``
and ship FP8 attention / dense MLP weights with MXFP4 routed experts.
Registration maps that config onto :class:`MiMoV2Config`, routes it to the
vision-language class, and registers the load-time conversions in
:mod:`agilerl.architectures.mimo_v2.checkpoint`. MTP layers and the audio
tower are never built and their keys are ignored. Idempotent; safe to call
twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.architectures.runtime import PatchRuntimeConfig

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from transformers import AutoConfig
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    )
    from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
        MiMoV2FlashPreTrainedModel,
    )

    from agilerl.architectures.mimo_v2.checkpoint import (
        mimo_v2_checkpoint_conversions,
    )
    from agilerl.architectures.mimo_v2.configuration import (
        MIMO_V2_MODEL_TYPE,
        MiMoV2Config,
    )
    from agilerl.architectures.mimo_v2.layers import (
        install_mimo_v2_fp32_lse_flex_attention,
    )
    from agilerl.architectures.mimo_v2.modeling import MiMoV2ForConditionalGeneration
else:
    AutoConfig: Any = None
    register_checkpoint_conversion_mapping: Any = None
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING: Any = None
    MiMoV2FlashPreTrainedModel: Any = None
    install_mimo_v2_fp32_lse_flex_attention: Any = None
    mimo_v2_checkpoint_conversions: Any = None
    MIMO_V2_MODEL_TYPE: Any = None
    MiMoV2Config: Any = None
    MiMoV2ForConditionalGeneration: Any = None

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedModel

__all__ = [
    "install_mimo_v2_patches",
    "register_mimo_v2_native_loading",
]


def register_mimo_v2_native_loading() -> None:
    """Register the config, vision-language class and checkpoint conversions a native ``mimo_v2`` load needs."""
    if not HAS_LLM_DEPENDENCIES:
        return

    AutoConfig.register(MIMO_V2_MODEL_TYPE, MiMoV2Config, exist_ok=True)
    # The mapping types values as tuples but stores single classes everywhere.
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING.register(
        MiMoV2Config, cast("Any", MiMoV2ForConditionalGeneration), exist_ok=True
    )
    register_checkpoint_conversion_mapping(
        MIMO_V2_MODEL_TYPE, mimo_v2_checkpoint_conversions(), overwrite=True
    )


def install_mimo_v2_patches(
    _patch: PatchRuntimeConfig,
    _model: PreTrainedModel | PeftModel | None = None,
    _use_sequence_packing: bool = False,
) -> None:
    """Register native ``mimo_v2`` loading and flex attention with sinks on the fp32 log-sum-exp.

    Trainer-only: vLLM must read the Hub config, which registration replaces.

    :param _patch: Catalog patch config.
    :type _patch: PatchRuntimeConfig
    :param _model: Already-built model, or None; the flag is read at load.
    :type _model: PreTrainedModel | PeftModel | None
    :param _use_sequence_packing: Unused; MiMo-V2 has no packed-sequence wrap.
    :type _use_sequence_packing: bool, optional
    """
    register_mimo_v2_native_loading()
    # The native class opts out of flex for its 192 / 128 head dims, not a
    # kernel limit; flex takes the sinks as ``s_aux`` on CUDA only.
    MiMoV2FlashPreTrainedModel._supports_flex_attn = True
    install_mimo_v2_fp32_lse_flex_attention()
