#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import sys

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmis.config import load_config, resolve_project_path
from tmis.utils import finish_run_state, initialize_run_state, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/twitter2015.yaml")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Override config output_dir; use a unique directory for each run.",
    )
    ap.add_argument(
        "--resume",
        default=None,
        help="Resume from outputs/.../latest.pt (model, optimizer, scheduler, and AMP state).",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Stable identifier recorded in run_state.json and exported reports.",
    )
    args = ap.parse_args()
    # Delay heavyweight Transformers imports so --help and config-facing CLI
    # diagnostics still work before server dependencies are installed.
    from tmis.runtime import (
        build_collator,
        build_datasets,
        build_model,
        build_tokenizer_and_processor,
        choose_device,
    )
    from tmis.training import StageTrainer

    cfg = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    output_value = args.output_dir if args.output_dir is not None else cfg["output_dir"]
    cfg["output_dir"] = str(resolve_project_path(cfg, output_value))
    run_id = args.run_id or f"{cfg['experiment_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None:
        raise ValueError("--run-id may contain only letters, digits, dot, underscore, and hyphen")
    cfg["_run_id"] = run_id
    set_seed(int(cfg["seed"]))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process training currently requires CUDA GPUs")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        backend = str(cfg["training"].get("distributed_backend", "nccl"))
        # Stage-3 autoregressive validation runs on rank 0 and can be lengthy;
        # keep worker barriers alive while it evaluates the full dev split.
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(hours=6),
        )
        device = torch.device("cuda", local_rank)
    else:
        device = choose_device(cfg["device"])

    is_main = not distributed or dist.get_rank() == 0
    run_state_path = None
    run_state = None
    if is_main:
        run_state_path, run_state = initialize_run_state(
            cfg["output_dir"],
            run_id=run_id,
            experiment_name=str(cfg["experiment_name"]),
            config_path=str(args.config),
            project_root=cfg["_project_root"],
            command=list(sys.argv),
        )

    try:
        tokenizer, image_processor = build_tokenizer_and_processor(cfg)
        train_ds, dev_ds, _ = build_datasets(cfg)
        collator = build_collator(cfg, tokenizer, image_processor)
        model = build_model(cfg, tokenizer)
        trainer = StageTrainer(cfg, model, tokenizer, train_ds, dev_ds, collator, device)
        trainer.train_all(args.resume)
        if is_main and run_state_path is not None and run_state is not None:
            finish_run_state(run_state_path, run_state, status="completed")
    except BaseException as exc:
        if is_main and run_state_path is not None and run_state is not None:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            try:
                finish_run_state(
                    run_state_path,
                    run_state,
                    status=status,
                    error={"type": type(exc).__name__, "message": str(exc)},
                )
            except Exception as metadata_error:
                print(
                    f"warning: failed to update run_state.json: {metadata_error}",
                    file=sys.stderr,
                )
        raise
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
