import torch

torch.set_printoptions(precision=4, sci_mode=False)

gamma = 1.0
lam = 0.95
beta = 0.05
score = 5.0
INVALID_LOGPROB = 1.0

# Full sequence: [p1, p2, tok1, tok2, tok3, eos, PAD, PAD]
# Prompt length = 2, response = 4 real tokens + 2 PAD
S = 8
context_length = 2
response_len = S - context_length  # 6

# --- Shared raw data (from the model) ---
# Values from critic (full sequence)
full_values_raw = torch.tensor([[1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 0.5, 0.3]])

# Log-probs: log P(token_{j+1} | tokens_{0..j}), shape (1, S-1=7)
#            index:  0(p2) 1(tok1) 2(tok2) 3(tok3) 4(eos)  5(PAD) 6(PAD)
old_lp = torch.tensor([[-2.0, -1.2, -0.8, -1.5, -0.3, -0.7, -0.4]])
ref_lp = torch.tensor([[-1.8, -1.0, -0.9, -1.2, -0.5, -0.6, -0.3]])

print("=" * 80)
print("REFERENCE IMPLEMENTATION (response-only tensors)")
print("=" * 80)

# Extract response-only slices (positions context_length-1 to S-2)
ref_logprobs = old_lp[
    :, context_length - 1 : -1
].clone()  # shape (1, 6) - skipping index 0 (prompt)
ref_ref_logprobs = ref_lp[:, context_length - 1 : -1].clone()
ref_values_raw = full_values_raw[:, context_length - 1 : -1].clone()  # (1, 6)

# Masks
seq_len_val = 3  # last non-pad response token = eos at response position 3
seq_len_p1 = seq_len_val + 1  # 4
resp_idxs = torch.arange(response_len).unsqueeze(0)
padding_mask = resp_idxs > seq_len_val  # [F,F,F,F,T,T]
padding_mask_p1 = resp_idxs > seq_len_p1  # [F,F,F,F,F,T]

# Mask fill
ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
ref_ref_logprobs = torch.masked_fill(ref_ref_logprobs, padding_mask, INVALID_LOGPROB)
ref_values = torch.masked_fill(ref_values_raw, padding_mask_p1, 0.0)

print("\nResponse tokens:    [tok1, tok2, tok3, eos, PAD, PAD]")
print("Response positions:  0     1     2     3    4    5")
print(f"padding_mask:       {(~(~padding_mask)).int().squeeze().tolist()}")
print(f"padding_mask_p1:    {(~(~padding_mask_p1)).int().squeeze().tolist()}")
print(f"logprobs:           {ref_logprobs.squeeze().tolist()}")
print(f"ref_logprobs:       {ref_ref_logprobs.squeeze().tolist()}")
print(f"values:             {ref_values.squeeze().tolist()}")

# KL and rewards
ref_kl = ref_logprobs - ref_ref_logprobs
ref_non_score = -beta * ref_kl
ref_rewards = ref_non_score.clone()
ref_rewards[0, seq_len_p1] += score  # score at position 4

print(f"\nKL per token:       {ref_kl.squeeze().tolist()}")
print(f"KL rewards:         {ref_non_score.squeeze().tolist()}")
print(f"Score at pos {seq_len_p1}:     +{score}")
print(f"Total rewards:      {ref_rewards.squeeze().tolist()}")

# GAE
ref_adv = torch.zeros_like(ref_rewards)
last_gae = torch.zeros(1)
for t in reversed(range(response_len)):
    nextval = ref_values[0, t + 1] if t < response_len - 1 else 0.0
    delta = ref_rewards[0, t] + gamma * nextval - ref_values[0, t]
    last_gae = delta + gamma * lam * last_gae
    ref_adv[0, t] = last_gae

ref_returns = ref_adv + ref_values

print(f"\nGAE advantages:     {ref_adv.squeeze().tolist()}")
print(f"Returns:            {ref_returns.squeeze().tolist()}")

# What the policy loss uses (masked by ~padding_mask)
active = ~padding_mask.squeeze()
print(f"\nPolicy uses positions {torch.where(active)[0].tolist()}")
print(f"  advantages:       {ref_adv.squeeze()[active].tolist()}")
print(f"  log_probs:        {ref_logprobs.squeeze()[active].tolist()}")
print("  Each log_prob[i] predicts response token i")

# Value loss uses (~padding_mask_p1)
active_v = ~padding_mask_p1.squeeze()
print(f"Value loss uses positions {torch.where(active_v)[0].tolist()}")
print(f"  returns:          {ref_returns.squeeze()[active_v].tolist()}")


print("\n" + "=" * 80)
print("YOUR IMPLEMENTATION (full-sequence, [:, :-1] KL alignment)")
print("=" * 80)

# action_mask: NOT shifted ([:, 1:] commented out in base.py)
action_mask = torch.tensor([[False, False, True, True, True, True, False, False]])

# Values masked
your_values = torch.masked_fill(full_values_raw.clone(), ~action_mask, 0.0)

# Log-probs masked with aligned_action_masks
aligned_mask = action_mask[:, 1:]  # (1, 7): [F, T, T, T, T, F, F]
your_old_lp = torch.masked_fill(old_lp.clone(), ~aligned_mask, INVALID_LOGPROB)
your_ref_lp = torch.masked_fill(ref_lp.clone(), ~aligned_mask, INVALID_LOGPROB)

print("\nFull sequence:      [p1,   p2,   tok1, tok2, tok3, eos,  PAD,  PAD]")
print("Full positions:      0     1     2     3     4     5     6     7")
print(f"action_mask:        {action_mask.int().squeeze().tolist()}")
print(f"values:             {your_values.squeeze().tolist()}")

# token_kl
your_kl = your_old_lp - your_ref_lp  # (1, 7)

# token_rewards: score at last action position (eos = position 5)
your_token_rewards = torch.zeros(1, S)
reward_pos = 5  # last True in action_mask
your_token_rewards[0, reward_pos] = score

# Apply KL with [:, :-1]
your_rewards = your_token_rewards.clone()
your_rewards[:, :-1] -= beta * your_kl

print(f"\naligned_mask [:, 1:]:{aligned_mask.int().squeeze().tolist()}")
print(f"old_log_probs:      {your_old_lp.squeeze().tolist()}")
print(f"ref_log_probs:      {your_ref_lp.squeeze().tolist()}")
print(f"token_kl (S-1):     {your_kl.squeeze().tolist()}")
print(f"beta*kl (S-1):      {(beta * your_kl).squeeze().tolist()}")
print(f"token_rewards:      {your_token_rewards.squeeze().tolist()}")
print(f"penalised_rewards:  {your_rewards.squeeze().tolist()}")

# GAE
your_adv = torch.zeros(1, S)
last_gae = torch.zeros(1)
print("\nGAE step-by-step:")
for t in reversed(range(S)):
    mask_t = action_mask[0, t]
    if t + 1 < S:
        nextval = your_values[0, t + 1] * action_mask[0, t + 1]
    else:
        nextval = 0.0
    delta = your_rewards[0, t] + gamma * nextval - your_values[0, t]
    last_gae = (delta + gamma * lam * last_gae) * mask_t
    your_adv[0, t] = last_gae
    if t <= 5 and t >= 1:
        print(
            f"  t={t} mask={int(mask_t)} r={your_rewards[0, t]:.4f} V={your_values[0, t]:.1f} "
            f"nextV={float(nextval):.1f} delta={float(delta):.4f} adv={float(last_gae):.4f}"
        )

your_returns = your_adv + your_values

print(f"\nadvantages:         {your_adv.squeeze().tolist()}")
print(f"returns:            {your_returns.squeeze().tolist()}")

# Training alignment
your_aligned_adv = your_adv[:, :-1]  # (1, 7)
your_aligned_ret = your_returns[:, :-1]  # (1, 7)

print("\n--- Training loop alignment ---")
print("log_probs index:    0     1     2     3     4     5     6")
print("predicts token at:  pos1  pos2  pos3  pos4  pos5  pos6  pos7")
print("                    (p2) (tok1)(tok2)(tok3)(eos) (PAD) (PAD)")
print(f"aligned_mask:       {aligned_mask.int().squeeze().tolist()}")
print(
    f"aligned_adv[:,:-1]: {[f'{x:.4f}' for x in your_aligned_adv.squeeze().tolist()]}"
)
print(
    f"aligned_ret[:,:-1]: {[f'{x:.4f}' for x in your_aligned_ret.squeeze().tolist()]}"
)

active_your = aligned_mask.squeeze().bool()
active_idxs = torch.where(active_your)[0].tolist()
print(f"\nPolicy loss uses indices {active_idxs}")
for i in active_idxs:
    tok_name = ["p2", "tok1", "tok2", "tok3", "eos", "PAD", "PAD"][i]
    print(f"  idx={i}: log P({tok_name}) * adv={your_aligned_adv[0, i]:.4f}")


print("\n" + "=" * 80)
print("PROBLEM: KL for first response token (tok1) lands at prompt position")
print("=" * 80)
print(f"""
With [:, :-1] alignment:
  - token_kl[1] = KL for generating tok1 = {your_kl[0, 1]:.2f}
  - beta * kl = {beta * your_kl[0, 1]:.4f}
  - This goes into rewards[1] = {your_rewards[0, 1]:.4f}  (position 1 = prompt!)
  - GAE at position 1: mask_t = F → advantage = 0.0 → KL reward is LOST

  - At training index 1 (generating tok1):
    aligned_adv[1] = advantages[1] = 0.0  (no gradient signal!)

Reference comparison:
  - rewards[0] = {ref_rewards[0, 0]:.4f}  (same KL, at response position 0)
  - advantage[0] = {ref_adv[0, 0]:.4f}  (full gradient signal)
""")

print("=" * 80)
print("SIDE-BY-SIDE: Active advantages used in policy loss")
print("=" * 80)
print(f"\n{'Action':<15} {'Reference adv':<18} {'Yours adv':<18} {'Diff':<10}")
print("-" * 61)
ref_active = ref_adv.squeeze()[~padding_mask.squeeze()].tolist()
# Map: ref pos 0 = generate tok1, ref pos 1 = generate tok2, etc.
action_names = ["gen tok1", "gen tok2", "gen tok3", "gen eos"]
your_active = [your_aligned_adv[0, i].item() for i in active_idxs]
for i, name in enumerate(action_names):
    r = ref_active[i] if i < len(ref_active) else 0.0
    y = your_active[i] if i < len(your_active) else 0.0
    print(f"{name:<15} {r:<18.4f} {y:<18.4f} {y - r:<10.4f}")

print(f"\n{'Return':<15} {'Reference':<18} {'Yours':<18}")
print("-" * 51)
ref_ret_active = ref_returns.squeeze()[~padding_mask.squeeze()].tolist()
your_ret_vals = [your_aligned_ret[0, i].item() for i in active_idxs]
for i, name in enumerate(action_names):
    r = ref_ret_active[i] if i < len(ref_ret_active) else 0.0
    y = your_ret_vals[i] if i < len(your_ret_vals) else 0.0
    print(f"{name:<15} {r:<18.4f} {y:<18.4f}")
