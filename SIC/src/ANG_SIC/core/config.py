import os
import json
import argparse
from typing import Optional, Dict, Any, Tuple, List

def ensure_parent_dir(path: str) -> None:
    if not path:
        return
    target = os.path.dirname(path) if os.path.splitext(path)[1] else path
    if target and not os.path.exists(target):
        os.makedirs(target, exist_ok=True)

def resolve_in_dir(maybe_path: Optional[str], base_dir: str) -> Optional[str]:
    if maybe_path is None:
        return None
    maybe_path = os.path.expanduser(os.path.expandvars(maybe_path))
    return maybe_path if os.path.isabs(maybe_path) else os.path.join(base_dir, maybe_path)

def build_output_path(suggested: Optional[str], default_dir: str, default_name: str) -> str:
    if suggested is None:
        path = os.path.join(default_dir, default_name)
    else:
        s = os.path.expanduser(os.path.expandvars(suggested))
        is_dir = s.endswith(("/", "\\")) or (os.path.exists(s) and os.path.isdir(s))
        path = os.path.join(s, default_name) if is_dir else s
    ensure_parent_dir(path)
    return path

def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

class LogBuffer:
    def __init__(self, path: str, flush_every: int = 1000):
        self.path = path
        self.buf = []
        self.flush_every = int(flush_every)
        self._disabled = (not path) or (str(path).strip().lower() in {"none", "null", ""})

    def add(self, obj: dict) -> None:
        if self._disabled:
            return
        self.buf.append(obj)
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._disabled or not self.buf:
            return
        ensure_parent_dir(self.path)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(map(json.dumps, self.buf)) + "\n")
        self.buf.clear()

def _load_file(path: str) -> Dict[str, Any]:
    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(path):
        return {}
    ext = os.path.splitext(path)[1].lower()
    if ext in {".yml", ".yaml"}:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    return {}

def load_sic_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sic_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = {
        "device": "auto",
        "model": "SmallMNISTCNN",
        "dataset": "MNIST",
        "data": {"batch_size": 512, "num_workers": 2, "pin_memory": True, "val_split": 0.0},
        "train": {"epochs": 5, "lr": 1e-3},
        "sic": {
            "mode": "hybrid",
            "visualize": False,
            "autosave_json_each_layer": True,
            "merge_after_clustering": False,
            "rounding_decimals": 6,
            "hybrid_enable": True,
            "row_len_gpu_threshold": 2048,
            "layer_params_gpu_threshold": 1_000_000,
            "eval_max": None,
            "max_passes": 3,
            "repeat_until_no_change": True,
            "enable_product_of_sums": False,
            "fast_pos_accel": True,
            "fast_pos_compile": True,
        },
        "io": {
            "models_dir": "models",
            "profiling_dir": "profiling",
            "neuron_log_path": "SIC_per_neuron_times.jsonl",
            "autosave_progress_path": "SIC_Progress_Autosave.json",
            "excel_path": None,
            "json_stats_path": None,
            "custom_load_path": None,
            "custom_save_name": None,
        },
    }

    file_cfg = _load_file(args.config)
    cfg = deep_update(cfg, file_cfg)

    if args.device:
        cfg["device"] = args.device
    if args.model:
        cfg["model"] = args.model
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.weights:
        cfg["io"]["custom_load_path"] = args.weights
    if args.save_name:
        cfg["io"]["custom_save_name"] = args.save_name

    return cfg

def load_ang_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="ang.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--src_layer", type=str, default=None)
    parser.add_argument("--destination_activation", type=str, default=None, choices=["relu", "tanh", "elu"])
    parser.add_argument("--elu_alpha", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", type=str, default=None)
    parser.add_argument("--prime_epochs", type=int, default=None)
    parser.add_argument("--grow_max_epochs", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--target_acc", type=float, default=None)
    parser.add_argument("--per_class_target", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--max_classes_per_cycle", type=int, default=None)
    parser.add_argument("--freeze_seed_after_priming", action="store_true")
    parser.add_argument("--es_patience", type=int, default=None)
    parser.add_argument("--es_target_val_acc", type=float, default=None)
    parser.add_argument("--models_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    args, _ = parser.parse_known_args()

    cfg = {
        "device": "auto",
        "seed": 0,
        "dataset": "MNIST",
        "model": "SeedCNN",
        "destination_activation": "relu",
        "elu_alpha": 1.0,
        "ang": {
            "src_layer": "conv2",
            "prime_epochs": 11,
            "grow_max_epochs": 50,
            "cycles": 18,
            "target_acc": 0.98,
            "per_class_target": 0.98,
            "scale": 1.0,
            "max_classes_per_cycle": None,
            "freeze_seed_after_priming": False,
            "es_patience": 20,
            "es_target_val_acc": 1.0,
        },
        "data": {"batch_size": 128, "num_workers": 4, "pin_memory": True},
        "io": {
            "models_dir": "models",
            "results_dir": "results",
            "custom_load_path": None,
            "custom_save_name": "grown_model.pth",
        },
    }

    if os.path.exists(args.config):
        ext = os.path.splitext(args.config)[1].lower()
        if ext in {".yml", ".yaml"}:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = deep_update(cfg, yaml.safe_load(f) or {})
        elif ext == ".json":
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = deep_update(cfg, json.load(f) or {})

    if args.device:
        cfg["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.model:
        cfg["model"] = args.model
    if args.src_layer:
        cfg["ang"]["src_layer"] = args.src_layer
    if args.destination_activation:
        cfg["destination_activation"] = args.destination_activation.lower()
    if args.elu_alpha is not None:
        cfg["elu_alpha"] = float(args.elu_alpha)
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.pin_memory is not None:
        cfg["data"]["pin_memory"] = str(args.pin_memory).lower() == "true"
    if args.prime_epochs is not None:
        cfg["ang"]["prime_epochs"] = args.prime_epochs
    if args.grow_max_epochs is not None:
        cfg["ang"]["grow_max_epochs"] = args.grow_max_epochs
    if args.cycles is not None:
        cfg["ang"]["cycles"] = args.cycles
    if args.target_acc is not None:
        cfg["ang"]["target_acc"] = args.target_acc
    if args.per_class_target is not None:
        cfg["ang"]["per_class_target"] = args.per_class_target
    if args.scale is not None:
        cfg["ang"]["scale"] = args.scale
    if args.max_classes_per_cycle is not None:
        cfg["ang"]["max_classes_per_cycle"] = args.max_classes_per_cycle
    if args.freeze_seed_after_priming:
        cfg["ang"]["freeze_seed_after_priming"] = True
    if args.es_patience is not None:
        cfg["ang"]["es_patience"] = args.es_patience
    if args.es_target_val_acc is not None:
        cfg["ang"]["es_target_val_acc"] = args.es_target_val_acc
    if args.models_dir:
        cfg["io"]["models_dir"] = args.models_dir
    if args.results_dir:
        cfg["io"]["results_dir"] = args.results_dir
    if args.weights:
        cfg["io"]["custom_load_path"] = args.weights
    if args.save_name:
        cfg["io"]["custom_save_name"] = args.save_name

    act = str(cfg.get("destination_activation", "relu")).lower()
    if act not in {"relu", "tanh", "elu"}:
        cfg["destination_activation"] = "relu"
    if cfg["destination_activation"] == "elu":
        try:
            cfg["elu_alpha"] = float(cfg.get("elu_alpha", 1.0))
        except Exception:
            cfg["elu_alpha"] = 1.0

    return cfg
