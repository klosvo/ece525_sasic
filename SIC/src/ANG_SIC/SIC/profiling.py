from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime
from time import perf_counter
from typing import Deque, Dict, Any, List, Optional

import json
import numpy
import psutil
from pathlib import Path

class MemoryTracker:
    def __init__(self) -> None:
        self.process = psutil.Process() if psutil is not None else None
        self.checkpoints: Dict[str, float] = {}
        self.timeline: List[Dict[str, Any]] = []
        self.start_memory: float = 0.0

    def start_tracking(self) -> None:
        self.start_memory = self.get_current_memory()
        self.checkpoint("start")

    def get_current_memory(self) -> float:
        if self.process is None:
            return 0.0
        return self.process.memory_info().rss / (1024 * 1024)

    def checkpoint(self, name: str) -> None:
        current = self.get_current_memory()
        self.checkpoints[name] = current
        self.timeline.append({"timestamp": perf_counter(), "checkpoint": name, "memory_mb": current})

    def get_peak_memory(self) -> float:
        return max((e["memory_mb"] for e in self.timeline), default=0.0)

    def get_memory_saved(self) -> float:
        if len(self.timeline) >= 2:
            return self.timeline[0]["memory_mb"] - self.timeline[-1]["memory_mb"]
        return 0.0

class SICProfiler:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.stats: Dict[str, Any] = {
            "global": {
                "start_time": None,
                "end_time": None,
                "total_duration": 0.0,
                "original_params": 0,
                "final_params": 0,
                "effective_params": 0,
                "zero_params": 0,
                "compression_ratio": 0.0,
                "memory_peak_mb": 0.0,
                "memory_saved_mb": 0.0,
            },
            "phases": {
                "initialization": {"duration": 0.0},
                "filtering": {"duration": 0.0, "samples_before": 0, "samples_after": 0},
                "clustering": {"duration": 0.0, "total_neurons": 0, "successful_neurons": 0},
                "merging": {"duration": 0.0, "merges_performed": 0},
                "verification": {"duration": 0.0, "checks_performed": 0},
            },
            "layers": {},
            "convergence": {
                "clustering_attempts": [],
                "failure_reasons": defaultdict(int),
                "efficiency_metrics": {},
            },
            "timing_breakdown": defaultdict(list),
            "memory_timeline": [],
        }
        self.timing_aggs: Dict[str, Dict[str, Any]] = {
            "jenks": {"total": 0.0, "count": 0, "samples": deque(maxlen=1000)},
            "all_samples_correct": {"total": 0.0, "count": 0, "samples": deque(maxlen=1000)},
        }
        self.memory_tracker: Optional[MemoryTracker] = None

    def _add_time(self, key: str, seconds: float) -> None:
        agg = self.timing_aggs.setdefault(key, {"total": 0.0, "count": 0, "samples": deque(maxlen=1000)})
        agg["total"] += float(seconds)
        agg["count"] += 1
        samples_deque: Deque[float] = agg["samples"]
        samples_deque.append(float(seconds))

    @staticmethod
    def _p95(samples: Deque[float]) -> float:
        if not samples:
            return 0.0
        arr = sorted(samples)
        idx = max(0, min(len(arr) - 1, int(numpy.ceil(0.95 * len(arr))) - 1))
        return float(arr[idx])

    def start_profiling(self, memory_tracker: Optional[MemoryTracker] = None) -> None:
        self.start_time = perf_counter()
        self.stats["global"]["start_time"] = datetime.now().isoformat()
        if memory_tracker is not None:
            self.memory_tracker = memory_tracker
        if self.memory_tracker is not None:
            self.memory_tracker.start_tracking()
        print(f"[PROFILER] Started profiling at {self.stats['global']['start_time']}")

    def end_profiling(self) -> None:
        end_time = perf_counter()
        self.stats["global"]["end_time"] = datetime.now().isoformat()
        self.stats["global"]["total_duration"] = (end_time - (self.start_time or end_time))
        if self.memory_tracker is not None:
            self.stats["global"]["memory_peak_mb"] = self.memory_tracker.get_peak_memory()
            self.stats["global"]["memory_saved_mb"] = self.memory_tracker.get_memory_saved()
            self.stats["memory_timeline"] = list(self.memory_tracker.timeline)
        
        # Compute final k-trials statistics (avg_k_trials_per_neuron)
        # (sasic_design.md §9: aggregate metrics for comparing baseline vs SASIC)
        if "sic" in self.stats:
            sic_stats = self.stats["sic"]
            if "total_k_trials" in sic_stats and "num_neurons_evaluated" in sic_stats:
                total_k = sic_stats["total_k_trials"]
                num_neurons = sic_stats["num_neurons_evaluated"]
                if num_neurons > 0:
                    sic_stats["avg_k_trials_per_neuron"] = float(total_k / num_neurons)
        
        print(f"[PROFILER] Completed profiling. Total duration: {self.stats['global']['total_duration']:.2f}s")

    def start_phase(self, phase_name: str) -> None:
        if phase_name not in self.stats["phases"]:
            self.stats["phases"][phase_name] = {"duration": 0.0}
        self.stats["phases"][phase_name]["start_time"] = perf_counter()
        if self.memory_tracker is not None:
            self.memory_tracker.checkpoint(f"start_{phase_name}")

    def end_phase(self, phase_name: str) -> None:
        phase = self.stats["phases"].get(phase_name)
        if phase and "start_time" in phase:
            duration = perf_counter() - phase["start_time"]
            self.stats["phases"][phase_name]["duration"] = duration
            if self.memory_tracker is not None:
                self.memory_tracker.checkpoint(f"end_{phase_name}")
            print(f"[PROFILER] Phase '{phase_name}' completed in {duration:.2f}s")

    def init_layer_stats(self, layer_name: str, weight_shape: tuple, total_params: int) -> None:
        self.stats["layers"][layer_name] = {
            "shape": tuple(weight_shape),
            "total_params": int(total_params),
            "neurons_processed": 0,
            "neurons_successful": 0,
            "neurons_failed": 0,
            "original_unique_weights": 0,
            "final_unique_weights": 0,
            "zero_weights_before": 0,
            "zero_weights_after": 0,
            "clustering_attempts": [],
            "processing_time": 0.0,
            "memory_usage_mb": 0.0,
            "weight_distribution": {
                "before": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
                "after": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
            },
        }

    def record_layer_processing(self, layer_name: str, processing_time: float, successful_neurons: int, total_neurons: int) -> None:
        if layer_name in self.stats["layers"]:
            failed = int(total_neurons) - int(successful_neurons)
            sr = float(successful_neurons) / max(1, int(total_neurons))
            self.stats["layers"][layer_name].update(
                {
                    "processing_time": float(processing_time),
                    "neurons_successful": int(successful_neurons),
                    "neurons_processed": int(total_neurons),
                    "neurons_failed": int(failed),
                    "success_rate": float(sr),
                }
            )

    def record_neuron_attempt(self, layer_name: str, neuron_idx: int, cluster_count: int, success: bool, reason: str = "") -> None:
        attempt = {
            "layer": layer_name,
            "neuron": int(neuron_idx),
            "clusters": int(cluster_count),
            "success": bool(success),
            "timestamp": perf_counter() - (self.start_time or 0.0),
            "reason": reason,
        }
        self.stats["convergence"]["clustering_attempts"].append(attempt)
        if not success:
            self.stats["convergence"]["failure_reasons"][reason] += 1
        if layer_name in self.stats["layers"]:
            self.stats["layers"][layer_name]["clustering_attempts"].append(attempt)

    def record_weight_distribution(self, layer_name: str, weights_before: Any, weights_after: Any) -> None:
        if layer_name not in self.stats["layers"]:
            return
        b = numpy.array(weights_before)
        a = numpy.array(weights_after)

        def _stats(x: numpy.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            return {"min": float(numpy.min(x)), "max": float(numpy.max(x)), "mean": float(numpy.mean(x)), "std": float(numpy.std(x))}

        self.stats["layers"][layer_name]["weight_distribution"]["before"] = _stats(b)
        self.stats["layers"][layer_name]["weight_distribution"]["after"] = _stats(a)
        self.stats["layers"][layer_name]["original_unique_weights"] = int(numpy.unique(b).size) if b.size else 0
        self.stats["layers"][layer_name]["final_unique_weights"] = int(numpy.unique(a).size) if a.size else 0
        self.stats["layers"][layer_name]["zero_weights_before"] = int(numpy.sum(b == 0)) if b.size else 0
        self.stats["layers"][layer_name]["zero_weights_after"] = int(numpy.sum(a == 0)) if a.size else 0

    def calculate_efficiency_metrics(self) -> None:
        attempts: List[Dict[str, Any]] = self.stats["convergence"]["clustering_attempts"]
        total = len(attempts)
        succ = sum(1 for a in attempts if a["success"])
        avg_clusters = (sum(a["clusters"] for a in attempts if a["success"]) / succ) if succ else 0.0
        self.stats["convergence"]["efficiency_metrics"] = {
            "total_attempts": int(total),
            "successful_attempts": int(succ),
            "overall_success_rate": float(succ / max(1, total)),
            "average_clusters_per_success": float(avg_clusters),
            "failure_distribution": dict(self.stats["convergence"]["failure_reasons"]),
        }

    def generate_report(self) -> str:
        self.calculate_efficiency_metrics()
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("SYNAPTIC INPUT CONSOLIDATION (SIC) PROFILING REPORT")
        lines.append("=" * 80)
        g = self.stats["global"]
        lines.append("\nGLOBAL STATISTICS")
        lines.append(f"Start Time: {g['start_time']}")
        lines.append(f"End Time: {g['end_time']}")
        lines.append(f"Total Duration: {g['total_duration']:.2f} seconds")
        lines.append(f"Peak Memory Usage: {g['memory_peak_mb']:.1f} MB")
        lines.append(f"Memory Saved: {g['memory_saved_mb']:.1f} MB")
        lines.append("\nPARAMETER ANALYSIS")
        lines.append(f"Original Parameters: {g['original_params']:,}")
        lines.append(f"Final Parameters: {g['final_params']:,}")
        lines.append(f"Effective (unique) Parameters: {g['effective_params']:,}")
        lines.append(f"Zero Parameters: {g['zero_params']:,}")
        lines.append(f"Compression Ratio: {g['compression_ratio']:.2f}x")
        lines.append("\nPHASE BREAKDOWN")
        for phase, data in self.stats["phases"].items():
            if "duration" in data:
                lines.append(f"{phase.capitalize()}: {data['duration']:.2f}s")
        lines.append("\nDETAILED TIMING (aggregated)")
        for key, agg in self.timing_aggs.items():
            mean = (agg["total"] / agg["count"]) if agg["count"] else 0.0
            p95 = self._p95(agg["samples"])
            lines.append(f"{key}: total={agg['total']:.4f}s, calls={agg['count']}, mean={mean:.6f}s, p95={p95:.6f}s")
        lines.append("\nLAYER-BY-LAYER ANALYSIS")
        for lname, ldata in self.stats["layers"].items():
            lines.append(f"\n  Layer: {lname}")
            lines.append(f"    Shape: {ldata.get('shape')}")
            lines.append(f"    Processing Time: {ldata.get('processing_time', 0.0):.2f}s")
            lines.append(f"    Success Rate: {ldata.get('success_rate', 0.0):.1%}")
            lines.append(f"    Unique Weights: {ldata.get('original_unique_weights',0)} → {ldata.get('final_unique_weights',0)}")
            lines.append(f"    Zero Weights: {ldata.get('zero_weights_before',0)} → {ldata.get('zero_weights_after',0)}")
        eff = self.stats["convergence"]["efficiency_metrics"]
        lines.append("\nCONVERGENCE ANALYSIS")
        lines.append(f"Total Clustering Attempts: {eff.get('total_attempts',0)}")
        lines.append(f"Successful Attempts: {eff.get('successful_attempts',0)}")
        lines.append(f"Overall Success Rate: {eff.get('overall_success_rate',0.0):.1%}")
        lines.append(f"Average Clusters per Success: {eff.get('average_clusters_per_success',0.0):.1f}")
        if eff.get("failure_distribution"):
            lines.append("\nFailure Reasons:")
            for reason, count in eff["failure_distribution"].items():
                lines.append(f"  {reason}: {count} times")
        lines.append("=" * 80)
        return "\n".join(lines)

    def save_detailed_stats(self, filepath: str) -> None:
        if not filepath:
            return
        from ..core.config import ensure_parent_dir
        ensure_parent_dir(filepath)
        Path(filepath).write_text(json.dumps(self._to_serializable(self.stats), indent=2), encoding="utf-8")

    def _to_serializable(self, obj: Any) -> Any:
        if isinstance(obj, defaultdict):
            obj = dict(obj)
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        return obj
