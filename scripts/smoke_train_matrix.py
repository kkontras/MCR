#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DatasetSuite:
    name: str
    release_dir: Path
    default_config: Path
    fold: int


@dataclass
class SmokeCase:
    suite: str
    method: str
    config_path: Path
    default_config_path: Path
    fold: int


def deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_case_config(case: SmokeCase, checkpoint_root: Path, batch_size: int) -> Dict:
    default_cfg = load_json(case.default_config_path)
    method_cfg = load_json(case.config_path)
    cfg = deep_merge(default_cfg, method_cfg)

    cfg.setdefault("training_params", {})
    cfg.setdefault("early_stopping", {})
    cfg.setdefault("model", {})
    cfg.setdefault("dataset", {})

    cfg["training_params"]["wandb_disable"] = True
    cfg["training_params"]["verbose"] = False
    cfg["training_params"]["tdqm_disable"] = True
    cfg["training_params"]["data_loader_workers"] = 0
    cfg["training_params"]["batch_size"] = int(batch_size)
    cfg["training_params"]["test_batch_size"] = int(batch_size)
    cfg["training_params"]["rec_test"] = False
    cfg["training_params"]["use_test_set"] = False

    cfg["early_stopping"]["max_epoch"] = 1
    cfg["early_stopping"]["n_steps_stop"] = 1
    cfg["early_stopping"]["n_steps_stop_after"] = 0
    cfg["early_stopping"]["validate_after"] = 0
    cfg["early_stopping"]["save_every_valstep"] = 10**9

    cfg["model"]["load_ongoing"] = False
    cfg["model"]["start_over"] = True
    cfg["model"]["save_base_dir"] = str(checkpoint_root / case.suite)
    if "save_dir" not in cfg["model"] or not cfg["model"]["save_dir"]:
        cfg["model"]["save_dir"] = f"smoke_{case.method}" + "_{}.pth.tar"

    if isinstance(cfg["dataset"].get("data_split"), dict):
        cfg["dataset"]["data_split"]["fold"] = case.fold
    cfg["dataset"]["fold"] = case.fold

    return cfg


def discover_cases(repo_root: Path, suites: List[DatasetSuite], selected: List[str]) -> List[SmokeCase]:
    selected_set = {x.lower() for x in selected}
    cases: List[SmokeCase] = []
    for suite in suites:
        if "all" not in selected_set and suite.name.lower() not in selected_set:
            continue
        if not suite.release_dir.exists():
            continue
        for cfg in sorted(suite.release_dir.glob("*.json")):
            cases.append(
                SmokeCase(
                    suite=suite.name,
                    method=cfg.stem,
                    config_path=cfg,
                    default_config_path=suite.default_config,
                    fold=suite.fold,
                )
            )
    return cases


def run_case(
    repo_root: Path,
    case: SmokeCase,
    case_config_path: Path,
    python_bin: str,
    timeout_sec: int,
    log_dir: Path,
) -> Tuple[bool, float, str]:
    cmd = [
        python_bin,
        "train.py",
        "--config",
        str(case_config_path),
        "--default_config",
        "None",
        "--fold",
        str(case.fold),
    ]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    elapsed = time.time() - started
    log_file = log_dir / f"{case.suite}_{case.method}.log"
    with log_file.open("w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n")
            f.write(proc.stderr)
    return proc.returncode == 0, elapsed, str(log_file)


def build_suites(repo_root: Path) -> List[DatasetSuite]:
    return [
        DatasetSuite(
            name="CREMA_D_res",
            release_dir=repo_root / "configs/CREMA_D/release/res",
            default_config=repo_root / "configs/CREMA_D/default_config_cremad_res.json",
            fold=0,
        ),
        DatasetSuite(
            name="CREMA_D_vit",
            release_dir=repo_root / "configs/CREMA_D/release/vit",
            default_config=repo_root / "configs/CREMA_D/default_config_cremad_vit.json",
            fold=0,
        ),
        DatasetSuite(
            name="AVE_res",
            release_dir=repo_root / "configs/AVE/release/res",
            default_config=repo_root / "configs/AVE/default_config_ave_res.json",
            fold=0,
        ),
        DatasetSuite(
            name="AVE_vit",
            release_dir=repo_root / "configs/AVE/release/vit",
            default_config=repo_root / "configs/AVE/default_config_ave_vit.json",
            fold=0,
        ),
        DatasetSuite(
            name="UCF_res",
            release_dir=repo_root / "configs/UCF/res",
            default_config=repo_root / "configs/UCF/default_config_ucf.json",
            fold=1,
        ),
        DatasetSuite(
            name="SthSth",
            release_dir=repo_root / "configs/SthSth/release",
            default_config=repo_root / "configs/SthSth/default_config_sthsth_2mod.json",
            fold=0,
        ),
        DatasetSuite(
            name="Mosei_VT",
            release_dir=repo_root / "configs/FactorCL/Mosei/release/VT",
            default_config=repo_root / "configs/FactorCL/Mosei/default_config_mosei_VT.json",
            fold=1,
        ),
        DatasetSuite(
            name="Mosei_VTA",
            release_dir=repo_root / "configs/FactorCL/Mosei/release/VTA",
            default_config=repo_root / "configs/FactorCL/Mosei/default_config_mosei_VTA.json",
            fold=1,
        ),
        DatasetSuite(
            name="Mosi_VT",
            release_dir=repo_root / "configs/FactorCL/Mosi/release/VT",
            default_config=repo_root / "configs/FactorCL/Mosi/default_config_mosi_VT.json",
            fold=0,
        ),
        DatasetSuite(
            name="Mosi_VTA",
            release_dir=repo_root / "configs/FactorCL/Mosi/release/VTA",
            default_config=repo_root / "configs/FactorCL/Mosi/default_config_mosi_VTA.json",
            fold=0,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-epoch smoke tests for all methods across datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Dataset suites to run (default: all).",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0 means no limit).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for smoke runs.")
    parser.add_argument("--timeout-sec", type=int, default=3600, help="Timeout per case in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Only print discovered cases.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop at first failure.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use for train.py (default: current python).",
    )
    parser.add_argument(
        "--output-json",
        default="smoke_results.json",
        help="Path to summary json report.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="",
        help="Override checkpoint root for smoke runs. Defaults to a temp dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    suites = build_suites(repo_root)
    cases = discover_cases(repo_root, suites, args.datasets)

    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        print("No cases discovered for selection:", args.datasets)
        return 1

    print(f"Discovered {len(cases)} smoke cases.")
    for case in cases:
        print(f"- {case.suite}: {case.method}")

    if args.dry_run:
        return 0

    checkpoint_root = Path(args.checkpoint_root).resolve() if args.checkpoint_root else Path(tempfile.mkdtemp(prefix="mcr_smoke_ckpt_"))
    tmp_cfg_dir = Path(tempfile.mkdtemp(prefix="mcr_smoke_cfg_"))
    log_dir = Path(tempfile.mkdtemp(prefix="mcr_smoke_logs_"))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint root: {checkpoint_root}")
    print(f"Temp config dir: {tmp_cfg_dir}")
    print(f"Logs dir: {log_dir}")

    results = []
    any_failed = False

    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] Running {case.suite}/{case.method}")
        case_cfg = build_case_config(case, checkpoint_root=checkpoint_root, batch_size=args.batch_size)
        case_cfg_path = tmp_cfg_dir / f"{case.suite}_{case.method}.json"
        with case_cfg_path.open("w", encoding="utf-8") as f:
            json.dump(case_cfg, f, indent=2)

        ok, elapsed, log_file = run_case(
            repo_root=repo_root,
            case=case,
            case_config_path=case_cfg_path,
            python_bin=args.python_bin,
            timeout_sec=args.timeout_sec,
            log_dir=log_dir,
        )
        print(f"  {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) log={log_file}")
        results.append(
            {
                "suite": case.suite,
                "method": case.method,
                "config": str(case.config_path),
                "default_config": str(case.default_config_path),
                "fold": case.fold,
                "ok": ok,
                "elapsed_sec": elapsed,
                "log_file": log_file,
            }
        )
        if not ok:
            any_failed = True
            if args.stop_on_fail:
                break

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "checkpoint_root": str(checkpoint_root),
        "temp_config_dir": str(tmp_cfg_dir),
        "logs_dir": str(log_dir),
        "results": results,
    }
    output_path = Path(args.output_json)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {output_path.resolve()}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
