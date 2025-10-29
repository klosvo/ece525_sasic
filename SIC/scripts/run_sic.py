import os
import copy
import hashlib
import numpy as np
import torch
import pandas as pd
import dill
import importlib.util
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

from src.ANG_SIC.core.config import (
    load_sic_config,
    build_output_path,
    ensure_parent_dir,
    resolve_in_dir,
)
from src.ANG_SIC.SIC.profiling import SICProfiler
from src.ANG_SIC.core.data import get_dataset_and_transform, resolve_loader_env
from src.ANG_SIC.models.models import build_model
from src.ANG_SIC.core.cleanup import collect_executed_module_names, strip_dead_zero_modules
from src.ANG_SIC.core.pipeline_utils import device_from_auto, maybe_materialize_pos
from src.ANG_SIC.SIC.verify import verify_model
from src.ANG_SIC.SIC.pruning import run_pruning_if_needed
from src.ANG_SIC.SIC.sic_alg import SIC, SIC_hybrid
from src.ANG_SIC.SIC.sic_utils import evaluate

def _make_clean_eval_loader(cfg: Dict[str, Any]) -> Tuple[torch.device, DataLoader]:
    device = device_from_auto(cfg.get("device", "auto"))
    data_cfg = cfg.get("data", {}) or {}
    bs = int(data_cfg.get("batch_size", 128))
    policy = resolve_loader_env(
        device=device,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )
    pin_memory = policy.pin_memory
    num_workers = policy.num_workers
    persistent = policy.persistent_workers
    worker_init_fn = policy.worker_init_fn
    sic_norm = bool(data_cfg.get("normalize_for_sic") or data_cfg.get("normalize", False))
    train_ds_full, test_ds_full = get_dataset_and_transform(
        cfg.get("dataset", "MNIST"),
        cfg.get("model"),
        normalize=sic_norm,
    )
    eval_test_loader = DataLoader(
        test_ds_full,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        worker_init_fn=worker_init_fn,
    )
    return device, eval_test_loader

def _split_train_val(train_ds, cfg, device):
    val_ratio = float(cfg.get("data", {}).get("val_split", 0.0) or 0.0)
    if val_ratio <= 0.0:
        return train_ds, None
    from torch.utils.data import random_split
    val_size = max(1, int(len(train_ds) * val_ratio))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(
        train_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 0))),
    )
    policy = resolve_loader_env(
        device=device,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )
    pin_memory = policy.pin_memory
    num_workers = policy.num_workers
    persistent = policy.persistent_workers
    worker_init_fn = policy.worker_init_fn
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        worker_init_fn=worker_init_fn,
    )
    return train_ds, val_loader

def main():
    cfg = load_sic_config()
    sic = cfg.setdefault("sic", {})
    if "use_hybrid_torch_gpu" in sic and "hybrid_enable" not in sic:
        sic["hybrid_enable"] = bool(sic.pop("use_hybrid_torch_gpu"))
    sic.setdefault("rounding_decimals", 6)
    sic.setdefault("enable_product_of_sums", False)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device_from_auto(cfg.get("device", "auto"))

    io = cfg.get("io", {}) or {}
    models_dir = io.get("models_dir", "models")
    profiling_dir = io.get("profiling_dir", "profiling")
    ensure_parent_dir(models_dir + "/")
    ensure_parent_dir(profiling_dir + "/")

    data_cfg = cfg.get("data", {}) or {}
    sic_norm = bool(data_cfg.get("normalize_for_sic") or data_cfg.get("normalize", False))
    train_ds, test_ds = get_dataset_and_transform(
        cfg.get("dataset", "MNIST"),
        cfg.get("model"),
        normalize=sic_norm,
    )
    train_ds, val_loader = _split_train_val(train_ds, cfg, device)

    policy = resolve_loader_env(
        device=device,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
    )
    pin_memory = policy.pin_memory
    num_workers = policy.num_workers
    persistent = policy.persistent_workers
    worker_init_fn = policy.worker_init_fn
    bs = int(cfg["data"].get("batch_size", 128))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        worker_init_fn=worker_init_fn,
    )

    device_eval, eval_test_loader = _make_clean_eval_loader(cfg)

    obj_path = io.get("custom_model_object_path")
    weights_path_cfg = io.get("custom_load_path")

    if obj_path and os.path.exists(obj_path):
        model = torch.load(obj_path, map_location=device, pickle_module=dill).to(device).eval()
        base_model_name = os.path.basename(obj_path)
        ensure_parent_dir(models_dir + "/")
        tmp_verify_path = os.path.join(models_dir, f"__tmp_verify_{base_model_name}.pth")
        torch.save(model.state_dict(), tmp_verify_path)
        weights_path = tmp_verify_path
    else:
        model = build_model(cfg.get("model"), cfg.get("dataset")).to(device)
        if weights_path_cfg:
            weights_path = resolve_in_dir(weights_path_cfg, models_dir)
        else:
            weights_path = os.path.join(models_dir, f"{cfg.get('model')}_{cfg.get('dataset')}.pth")
        base_model_name = os.path.basename(weights_path)
        if os.path.exists(weights_path):
            sd = torch.load(weights_path, map_location=device)
            model.load_state_dict(sd, strict=True)
        else:
            raise SystemExit("Model not trained")

    if weights_path_cfg:
        base_model_name = os.path.basename(resolve_in_dir(weights_path_cfg, models_dir))
    save_path = (
        resolve_in_dir(io.get("custom_save_name"), models_dir)
        if io.get("custom_save_name")
        else os.path.join(models_dir, f"SIC_{base_model_name}")
    )

    sic_base = os.path.splitext(os.path.basename(save_path))[0]
    excel_path = build_output_path(io.get("excel_path"), profiling_dir, f"{sic_base}.xlsx")
    enable_json = bool(io.get("enable_json_stats", True))
    enable_log = bool(io.get("enable_neuron_log", True))
    enable_auto = bool(io.get("enable_autosave_progress", True))
    json_path = (
        build_output_path(io.get("json_stats_path"), profiling_dir, f"{sic_base}_detailed_stats.json")
        if enable_json
        else None
    )
    neuron_log_path = build_output_path(
        io.get("neuron_log_path", os.path.join(profiling_dir, "SIC_per_neuron_times.jsonl")),
        profiling_dir,
        "SIC_per_neuron_times.jsonl",
    ) if enable_log else None
    autosave_progress_path = build_output_path(
        io.get("autosave_progress_path", os.path.join(profiling_dir, "SIC_Progress_Autosave.json")),
        profiling_dir,
        "SIC_Progress_Autosave.json",
    ) if enable_auto else None
    cfg["io"]["neuron_log_path"] = neuron_log_path
    cfg["io"]["autosave_progress_path"] = autosave_progress_path

    print("\nINITIAL MODEL ANALYSIS")
    pre_profiler = SICProfiler()
    ds = (cfg.get("dataset", "MNIST") or "").strip().lower()
    input_spec = ("conv", (3, 32, 32)) if ds == "cifar10" else ("conv", (1, 28, 28))
    verify_model(
        weights_path,
        copy.deepcopy(model).to("cpu").eval(),
        profiler=pre_profiler,
        rounding_decimals=int(cfg.get("sic", {}).get("rounding_decimals", 16)),
        input_spec=input_spec,
    )

    print("\nBASELINE ACCURACY")
    baseline_test_acc = float(evaluate(model.to(device_eval).eval(), eval_test_loader, device_eval))
    print(f"[EVAL] Test (before SIC): {baseline_test_acc:.2f}%")
    if val_loader is not None:
        baseline_val_acc = float(evaluate(model.to(device_eval).eval(), val_loader, device_eval))
        print(f"[EVAL] Val  (before SIC): {baseline_val_acc:.2f}%")

    def _fingerprint(m: torch.nn.Module) -> str:
        h = hashlib.sha256()
        for t in m.state_dict().values():
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    fp_before = _fingerprint(model)

    sic_mode = (cfg.get("sic", {}).get("mode") or "hybrid").lower()
    sic_profiler = SICProfiler()
    print("\nRUNNING SIC", sic_mode.upper())
    if sic_mode == "hybrid":
        model, _ = SIC_hybrid(
            model,
            train_loader,
            device,
            visualize=bool(cfg.get("sic", {}).get("visualize", False)),
            profiler=sic_profiler,
            cfg=cfg,
            val_loader=val_loader,
        )
    else:
        model, _ = SIC(
            model,
            train_loader,
            device,
            visualize=bool(cfg.get("sic", {}).get("visualize", False)),
            profiler=sic_profiler,
            cfg=cfg,
            val_loader=val_loader,
        )

    changed_by_sic = _fingerprint(model) != fp_before
    print(f"[SIC] Model changed: {changed_by_sic}")

    print("\nPRE-PRUNE ACCURACY")
    preprune_test_acc = float(evaluate(model.to(device_eval).eval(), eval_test_loader, device_eval))
    print(f"[EVAL] Test (pre-prune): {preprune_test_acc:.2f}%")
    if val_loader is not None:
        preprune_val_acc = float(evaluate(model.to(device_eval).eval(), val_loader, device_eval))
        print(f"[EVAL] Val  (pre-prune): {preprune_val_acc:.2f}%")

    prune_cfg = cfg.get("prune", {}) or {}
    if bool(prune_cfg.get("enabled", False)):
        prune_mode = str(prune_cfg.get("mode", "after"))
        if prune_mode == "after_zero" and changed_by_sic:
            print("[PRUNE] Skipped after_zero due to SIC changes")
        else:
            model, _ = run_pruning_if_needed(
                model,
                cfg,
                train_loader,
                val_loader,
                device,
                profiler=sic_profiler,
                reason=f"mode={prune_mode}",
            )

    model = maybe_materialize_pos(model, cfg.get("sic", {}) or {})

    sample = next(iter(eval_test_loader))
    if isinstance(sample, (tuple, list)) and len(sample) >= 1:
        sample_batch = sample[0]
    elif isinstance(sample, dict):
        sample_batch = (
            sample.get("pixel_values")
            or sample.get("images")
            or sample.get("input")
            or sample.get("x")
            or next(iter(sample.values()))
        )
    else:
        sample_batch = sample
    sample_batch = sample_batch.to(device)
    if hasattr(sample_batch, "size") and callable(sample_batch.size) and sample_batch.size(0) > 1:
        sample_batch = sample_batch[:1]

    with torch.inference_mode():
        executed = collect_executed_module_names(model, sample_batch)
    removed = strip_dead_zero_modules(model, executed, targets=None)
    if removed:
        print(f"[PRUNE] Removed dead modules: {removed}")

    ensure_parent_dir(save_path)
    torch.save(model.state_dict(), save_path)
    print(f"[SIC] Weights saved: {save_path}")

    print("\nFINAL MODEL ANALYSIS")
    final_profiler = SICProfiler()
    verify_model(
        save_path,
        copy.deepcopy(model).to("cpu").eval(),
        profiler=final_profiler,
        rounding_decimals=int(cfg.get("sic", {}).get("rounding_decimals", 6)),
        input_spec=input_spec,
    )

    print("\nFINAL ACCURACY")
    final_test_acc = float(evaluate(model.to(device_eval).eval(), eval_test_loader, device_eval))
    print(f"[EVAL] Test (final): {final_test_acc:.2f}%")
    if val_loader is not None:
        final_val_acc = float(evaluate(model.to(device_eval).eval(), val_loader, device_eval))
        print(f"[EVAL] Val  (final): {final_val_acc:.2f}%")

    if json_path:
        final_profiler.save_detailed_stats(json_path)

    if bool(cfg["io"].get("write_full_weights_to_excel", False)):
        ensure_parent_dir(excel_path)
        global_df = pd.DataFrame(
            [
                {
                    "start_time": final_profiler.stats["global"].get("start_time"),
                    "end_time": final_profiler.stats["global"].get("end_time"),
                    "total_duration_sec": final_profiler.stats["global"].get("total_duration", 0.0),
                    "memory_peak_mb": final_profiler.stats["global"].get("memory_peak_mb", 0.0),
                    "memory_saved_mb": final_profiler.stats["global"].get("memory_saved_mb", 0.0),
                }
            ]
        )
        has_openpyxl = importlib.util.find_spec("openpyxl") is not None
        if has_openpyxl:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as xls:
                global_df.to_excel(xls, index=False, sheet_name="global")
            print(f"[SIC] Excel report: {excel_path}")
        else:
            csv_path = os.path.splitext(excel_path)[0] + "_global.csv"
            global_df.to_csv(csv_path, index=False)
            print(f"[SIC] CSV report: {csv_path}")

    print("\nSIC PROFILER REPORT")
    print("[SIC] Done.")

if __name__ == "__main__":
    main()
