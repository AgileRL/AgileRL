# Plan: combine Context Parallel + Expert Parallel with mesh compose

**Worktree:** `/home/mike/wt-cp-ep`  
**Branch:** `migration/cp-ep` (from `migration/fsdp` @ `eda3f380`)  
**Sources:**

| Source | Path | Tip |
|--------|------|-----|
| FSDP base | `/home/mike/AgileRL` (`migration/fsdp`) | `eda3f380` |
| Context Parallel | `/home/mike/wt-context-parallel` (`migration/context-parallel`) | `78905086` |
| Expert Parallel | `/home/mike/wt-expert-parallel` (`migration/expert-parallel`) | `f8801fac` |

**Goal:** One branch where FSDP2 + CP and/or EP work, including **CP×EP together** via Prime-RL DeviceMesh composition.

Success gates:

1. FSDP + `CP=2` Ulysses (and ring) on `Qwen/Qwen2.5-0.5B`
2. FSDP + `EP=2` on hub MoE `PrimeIntellect/qwen3-moe-tiny`
3. **Joint** FSDP + `CP=2` + `EP=2` ConstantTarget learning smoke on 2×L4

Do **not** force-push `migration/fsdp` or rewrite the open FSDP PR unless explicitly requested.

---

## 1. Merge shape

Both CP and EP are fast-forward onto `migration/fsdp` alone. Combining them conflicts in 7 files (ctor kwargs + `apply_fsdp2` mesh vs `ep_mesh`). Resolve by **union API**, not pick-one.

Non-conflicting new modules stay as-is (`cp.py`, `ulysses_attn.py`, `ring_attn_compat.py`, `expert_parallel.py`, fixtures, tests, smoke docs).

---

## 2. Mesh compose (required)

Prime-RL rules (from EP plan):

```text
ep % cp == 0
(dp_shard * cp) % ep == 0
ep mesh = flatten(dp_shard_in_ep × cp)
```

| Mode | World dims | FSDP non-experts | FSDP experts | CP group | EP group |
|------|------------|------------------|--------------|----------|----------|
| flat `cp=1,ep=1` | none (PG) | flat PG | n/a | n/a | n/a |
| CP only | `(dp_shard, cp)` | `dp_shard_cp` flatten | n/a | `cp` | n/a |
| EP only | `(dp_mod_ep, dp_in_ep)` | `hsdp` | `dp_mod_ep` | n/a | `ep` |
| **CP×EP** | `(dp_mod_ep, dp_in_ep, cp)` with `dp_in_ep = ep // cp` | `hsdp` / `dp_shard_cp` flatten | `dp_mod_ep` | `cp` | flatten(`dp_in_ep × cp`) |

**2×L4 joint smoke** (`world_size=2`, `cp=2`, `ep=2`): `dp_mod_ep=1`, `dp_in_ep=1`, `cp=2` — EP and CP groups both span the two ranks.

### Validation

- `ep > 1` ⇒ FSDP + MoE + `num_experts % ep == 0` + world divisible by ep
- `cp > 1` ⇒ FSDP + FA2 / ring deps
- `cp > 1 and ep > 1` ⇒ `ep % cp == 0` and layout divisibility (not refused)

### Target API

```python
cp: int = 1
cp_style: Literal["ulysses", "ring"] = "ulysses"
ep: int = 1

def apply_fsdp2(model, config=None, *, mesh=None, ep_mesh=None) -> None: ...
# When both set: experts on ep_mesh.dp_mod_ep, non-experts on ep_mesh.hsdp
# (hsdp includes CP ranks); CP collectives use parallel_dims cp group.
```

Single planner (`ParallelDims` / `build_hybrid_parallel_mesh`) builds one world and exposes views for FSDP, CP, and EP.

---

## 3. Execution

1. FF EP (`f8801fac`), merge CP (`78905086`), resolve conflicts to union.
2. Implement hybrid mesh builder + composed `apply_fsdp2`.
3. Unit tests: layout `(W,cp,ep)` including `(2,2,2)` and `(8,2,4)`; keep CP/EP suites green.
4. GPU smokes (lock `/tmp/agilerl-smoke/gpu.lock.d`): CP-only → EP-only → joint CP×EP.
5. Write `docs/migration/smoke/RESULTS-cp-ep-combine.md`; local commits only.

---

## 4. Non-goals

- Force-push / rewrite FSDP PR `#642`
- vLLM colocate under CP×EP
- Worlds that violate `ep % cp == 0`

---

## 5. Status

| Item | State |
|------|--------|
| Worktree `/home/mike/wt-cp-ep` | Created |
| Branch `migration/cp-ep` | Active |
| Mutual exclusion of CP×EP | **Removed** — compose required |
| EP merge | Pending |
| CP merge | Pending |
| Hybrid mesh | Pending |
| Combine smokes | Pending |
