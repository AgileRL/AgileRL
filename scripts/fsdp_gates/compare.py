# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Compare paired DeepSpeed vs FSDP2 gate summaries against tolerances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(b), 1e-8)
    return abs(a - b) / denom


def compare_pair(
    ds: dict[str, Any],
    fsdp: dict[str, Any],
    *,
    loss_rel_mae: float = 0.05,
    eval_rel: float = 0.05,
    speed_floor: float = 0.90,
    vram_ceil: float = 1.10,
) -> list[str]:
    """Return a list of failure strings (empty ⇒ pass)."""
    failures: list[str] = []
    if ds.get("status") != "ok":
        failures.append(f"deepspeed status={ds.get('status')}")
    if fsdp.get("status") != "ok":
        failures.append(f"fsdp2 status={fsdp.get('status')}")

    if ds.get("final_loss") is not None and fsdp.get("final_loss") is not None:
        if _rel_err(float(fsdp["final_loss"]), float(ds["final_loss"])) > loss_rel_mae:
            failures.append(
                f"final_loss rel err "
                f"{_rel_err(float(fsdp['final_loss']), float(ds['final_loss'])):.3f} "
                f"> {loss_rel_mae}"
            )

    if ds.get("final_eval") is not None and fsdp.get("final_eval") is not None:
        if _rel_err(float(fsdp["final_eval"]), float(ds["final_eval"])) > eval_rel:
            failures.append(
                f"final_eval rel err "
                f"{_rel_err(float(fsdp['final_eval']), float(ds['final_eval'])):.3f} "
                f"> {eval_rel}"
            )

    ds_ms = ds.get("mean_step_ms")
    fs_ms = fsdp.get("mean_step_ms")
    if ds_ms and fs_ms and float(ds_ms) > 0:
        # Higher step_ms is slower; speed ratio = ds_ms / fs_ms
        speed_ratio = float(ds_ms) / float(fs_ms)
        if speed_ratio < speed_floor:
            failures.append(
                f"speed ratio {speed_ratio:.3f} < floor {speed_floor} "
                f"(ds_ms={ds_ms}, fsdp_ms={fs_ms})"
            )

    ds_vram = ds.get("peak_smi_used_mib") or ds.get("peak_allocated_bytes")
    fs_vram = fsdp.get("peak_smi_used_mib") or fsdp.get("peak_allocated_bytes")
    if ds_vram and fs_vram and float(ds_vram) > 0:
        ratio = float(fs_vram) / float(ds_vram)
        if ratio > vram_ceil:
            failures.append(f"vram ratio {ratio:.3f} > ceil {vram_ceil}")

    return failures


def find_summaries(job_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate deepspeed and fsdp2 summary.json under a job directory."""
    ds = list(job_dir.glob("deepspeed/**/summary.json"))
    fs = list(job_dir.glob("fsdp2/**/summary.json"))
    return (ds[0] if ds else None, fs[0] if fs else None)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts; empty list if missing."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _extract_curve(
    rows: list[dict[str, Any]],
    step_key: str = "train/global_step",
    loss_keys: tuple[str, ...] = ("train/loss", "loss", "train/mean_loss"),
) -> list[tuple[int, float]]:
    """Pull (step, loss) pairs from JSONL rows, trying multiple key names."""
    curve: list[tuple[int, float]] = []
    for row in rows:
        step = row.get(step_key)
        if step is None:
            for k in ("step", "global_step"):
                if k in row:
                    step = row[k]
                    break
        if step is None:
            continue
        loss = None
        for k in loss_keys:
            if k in row and row[k] is not None:
                loss = row[k]
                break
        if loss is None:
            continue
        try:
            curve.append((int(step), float(loss)))
        except (ValueError, TypeError):
            continue
    return curve


def compare_curves(
    ds_rows: list[dict[str, Any]],
    fsdp_rows: list[dict[str, Any]],
    *,
    max_rel_mae: float = 0.05,
    min_overlap: int = 5,
) -> dict[str, Any]:
    """Compare per-step loss curves between DeepSpeed and FSDP2 runs.

    Aligns the two curves by step index, computes per-step relative
    absolute error, and reports MAE, max rel err, and a pass/fail flag.

    :returns: Dict with keys ``pass``, ``mae``, ``max_rel_err``,
        ``overlap``, ``ds_points``, ``fsdp_points``.
    """
    ds_curve = _extract_curve(ds_rows)
    fs_curve = _extract_curve(fsdp_rows)

    result: dict[str, Any] = {
        "ds_points": len(ds_curve),
        "fsdp_points": len(fs_curve),
        "overlap": 0,
        "mae": None,
        "max_rel_err": None,
        "pass": True,
        "failures": [],
    }

    if not ds_curve or not fs_curve:
        result["pass"] = False
        result["failures"].append("missing curve data")
        return result

    ds_map = {s: v for s, v in ds_curve}
    fs_map = {s: v for s, v in fs_curve}
    common = sorted(set(ds_map) & set(fs_map))
    result["overlap"] = len(common)

    if len(common) < min_overlap:
        result["pass"] = False
        result["failures"].append(f"overlap {len(common)} < min {min_overlap}")
        return result

    rel_errs: list[float] = []
    for s in common:
        ds_v = ds_map[s]
        fs_v = fs_map[s]
        rel_errs.append(_rel_err(fs_v, ds_v))

    mae = sum(rel_errs) / len(rel_errs)
    max_rel = max(rel_errs)
    result["mae"] = mae
    result["max_rel_err"] = max_rel

    if mae > max_rel_mae:
        result["pass"] = False
        result["failures"].append(f"curve MAE {mae:.4f} > {max_rel_mae}")
    if max_rel > max_rel_mae * 3:
        result["pass"] = False
        result["failures"].append(
            f"curve max rel err {max_rel:.4f} > {max_rel_mae * 3:.4f}"
        )

    return result


def find_metrics_jsonl(job_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate deepspeed and fsdp2 train_metrics.jsonl under a job directory."""
    ds = list(job_dir.glob("deepspeed/**/train_metrics.jsonl"))
    fs = list(job_dir.glob("fsdp2/**/train_metrics.jsonl"))
    return (ds[0] if ds else None, fs[0] if fs else None)


def main(argv: list[str] | None = None) -> int:
    """CLI entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, help="Artifact dir for one job id")
    parser.add_argument("--loss-rel-mae", type=float, default=0.05)
    parser.add_argument("--eval-rel", type=float, default=0.05)
    parser.add_argument("--speed-floor", type=float, default=0.90)
    parser.add_argument("--vram-ceil", type=float, default=1.10)
    args = parser.parse_args(argv)

    ds_path, fs_path = find_summaries(args.job_dir)
    if ds_path is None or fs_path is None:
        print(f"Missing paired summaries under {args.job_dir}", file=sys.stderr)
        print(f"  deepspeed={ds_path} fsdp2={fs_path}", file=sys.stderr)
        return 2

    ds = _load_summary(ds_path)
    fs = _load_summary(fs_path)
    failures = compare_pair(
        ds,
        fs,
        loss_rel_mae=args.loss_rel_mae,
        eval_rel=args.eval_rel,
        speed_floor=args.speed_floor,
        vram_ceil=args.vram_ceil,
    )

    report: dict[str, Any] = {
        "job_dir": str(args.job_dir),
        "deepspeed_summary": str(ds_path),
        "fsdp2_summary": str(fs_path),
        "pass": not failures,
        "failures": failures,
    }

    ds_metrics_path, fs_metrics_path = find_metrics_jsonl(args.job_dir)
    if ds_metrics_path and fs_metrics_path:
        ds_rows = _load_jsonl(ds_metrics_path)
        fs_rows = _load_jsonl(fs_metrics_path)
        curve_result = compare_curves(ds_rows, fs_rows, max_rel_mae=args.loss_rel_mae)
        report["curve_comparison"] = curve_result
        report["deepspeed_metrics"] = str(ds_metrics_path)
        report["fsdp2_metrics"] = str(fs_metrics_path)
        if not curve_result["pass"]:
            report["pass"] = False
            report["failures"].extend(f"curve: {f}" for f in curve_result["failures"])
    else:
        report["curve_comparison"] = {"skipped": True}
    out = args.job_dir / "compare.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
