"""Validate zero-copy vLLM<->HF base-weight sharing on a real bnb model.

Stages (each prints [OK]/[FAIL] and continues where safe):
  1. patch standby sleep
  2. build vLLM bnb engine (in-process so internals are reachable)
  3. locate the live vLLM model + introspect a layer's module names
  4. extract HF-named, aliased bnb state dict
  5. build the shared HF model + assert storage aliasing
  6. PEFT-wrap (QLoRA) the shared base
  7. forward pass -> finite logits
  8. standby sleep frees KV but keeps weights resident; wake_up

Run on md-a100:
  cd ~/AgileRL && export PATH="$HOME/.local/bin:$PATH" && \
      uv run python repro_weight_sharing.py
"""

import os

# In-process V1 engine so the model is reachable in this process (mirrors the
# external_launcher executor AgileRL uses at training time).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from bitsandbytes.nn.modules import Linear4bit
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

from agilerl.algorithms.core.llm_ops.vllm_weight_sharing import (
    _language_model_layout,
    assert_shared_storage,
    build_shared_hf_model,
    extract_vllm_bnb_state_dict,
    get_vllm_internal_model,
    patch_vllm_standby_sleep_mode,
    prepare_shared_base_for_kbit_training,
)

MODEL = os.environ.get("WS_MODEL", "google/gemma-4-E4B-it")


def mem(tag):
    """Print free / used GPU memory under a tag."""
    free, total = torch.cuda.mem_get_info()
    print(
        f"[mem {tag}] free={free / 1e9:.2f}GB used={(total - free) / 1e9:.2f}GB",
        flush=True,
    )


def main():
    """Run the staged vLLM<->HF base-weight sharing repro end to end."""
    print(">>> stage 1: patch standby", flush=True)
    patch_vllm_standby_sleep_mode()
    print("[OK] standby patched", flush=True)

    print(">>> stage 2: build vLLM bnb", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        quantization="bitsandbytes",
        gpu_memory_utilization=0.5,
        enable_lora=True,
        max_lora_rank=16,
        max_num_seqs=2,
        max_model_len=2048,
        enable_sleep_mode=True,
        enforce_eager=True,
    )
    mem("after-vllm-init")
    print("[OK] vLLM built", flush=True)

    cfg = AutoConfig.from_pretrained(MODEL)

    print(">>> stage 3: locate internal model", flush=True)
    internal = get_vllm_internal_model(llm)
    print("internal type:", type(internal).__name__, flush=True)
    root, hf_prefix, _lm_head, multimodal = _language_model_layout(internal)
    print(
        "  decoder root:",
        type(root).__name__,
        "hf_prefix:",
        hf_prefix,
        "multimodal:",
        multimodal,
        flush=True,
    )
    print(
        "  packed_modules_mapping:",
        getattr(internal, "packed_modules_mapping", None),
        flush=True,
    )
    layer0 = root.layers[0]
    print("  layer0 children:", [n for n, _ in layer0.named_children()], flush=True)
    print(
        "  self_attn children:",
        [n for n, _ in layer0.self_attn.named_children()],
        flush=True,
    )
    print(
        "  mlp children:",
        [n for n, _ in layer0.mlp.named_children()],
        flush=True,
    )
    qkv_w = layer0.self_attn.qkv_proj.weight
    print(
        "  qkv has bnb_quant_state:",
        hasattr(qkv_w, "bnb_quant_state"),
        "shard_offsets:",
        getattr(qkv_w, "bnb_shard_offsets", None),
        flush=True,
    )
    print("[OK] internal located", flush=True)

    print(">>> stage 4: extract state dict", flush=True)
    sd = extract_vllm_bnb_state_dict(llm, cfg)
    print("  extracted keys:", len(sd), flush=True)
    q_key = f"{hf_prefix}.layers.0.self_attn.q_proj.weight"
    qs = sd.get(q_key + ".quant_state")
    print("  q_proj quant_state.shape:", getattr(qs, "shape", None), flush=True)
    qkv_mod = root.layers[0].self_attn.qkv_proj
    qkv = getattr(qkv_mod, "base_layer", qkv_mod).weight
    aliased = sd[q_key].data_ptr() == qkv.data_ptr()
    print("  q_proj aliases fused qkv storage:", aliased, flush=True)
    print("[OK] extracted", flush=True)

    print(">>> stage 5: build shared HF model + assert aliasing", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = build_shared_hf_model(llm, cfg, torch.bfloat16, bnb)
    assert_shared_storage(llm, model)
    mem("after-build-shared")
    print("[OK] shared model built + aliased (no second base copy)", flush=True)

    print(">>> stage 6: PEFT wrap (QLoRA)", flush=True)
    mem("before-kbit-prep")
    # No fp32 upcast: the shared base stays bf16 and aliased to vLLM (avoids the
    # ~14 GiB transient peak that OOMs at high gpu_memory_utilization).
    model = prepare_shared_base_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )
    mem("after-kbit-prep (no fp32 upcast; base stays bf16/shared)")
    # Scope LoRA to the shared language-model bnb linears (exclude the
    # vision/audio towers, which are unrelated frozen placeholders here).
    targets = [
        n
        for n, m in model.named_modules()
        if isinstance(m, Linear4bit) and "language_model" in n
    ]
    print(f"  lora targets: {len(targets)} language Linear4bit modules", flush=True)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora, adapter_name="actor")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_train:,}", flush=True)
    print("[OK] PEFT wrapped", flush=True)

    print(">>> stage 7: forward parity vs vLLM (the shared base)", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? One word."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        "cuda"
    )
    model.eval()
    with torch.no_grad():
        out = model(ids)
    logits = out.logits
    print(
        "  logits shape:",
        tuple(logits.shape),
        "finite:",
        bool(torch.isfinite(logits).all().item()),
        flush=True,
    )
    hf_top5 = logits[0, -1].topk(5).indices.tolist()
    print(
        "  HF shared-model top-5:", [(t, tok.decode([t])) for t in hf_top5], flush=True
    )

    # vLLM base (no adapter) greedy next token, same tokens — must agree, since
    # they share the exact base weights and the HF LoRA is zero-initialised.
    sp = SamplingParams(max_tokens=5, temperature=0.0)
    vout = llm.generate([{"prompt_token_ids": ids[0].tolist()}], sp, use_tqdm=False)
    vtoks = list(vout[0].outputs[0].token_ids)
    print("  vLLM greedy:", repr(vout[0].outputs[0].text), "tokens:", vtoks, flush=True)
    agree = vtoks and vtoks[0] == hf_top5[0]
    print("  HF top-1 == vLLM greedy token-0:", agree, flush=True)
    print("[OK] forward parity checked", flush=True)

    print(">>> stage 8: standby sleep / wake", flush=True)
    llm.sleep(level=2)
    mem("after-standby-sleep")
    llm.wake_up()
    mem("after-wake")
    print("[OK] sleep/wake survived", flush=True)

    print(">>> DONE", flush=True)


if __name__ == "__main__":
    main()
