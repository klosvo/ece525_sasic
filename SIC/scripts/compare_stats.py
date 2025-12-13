#!/usr/bin/env python3
"""
Compare two SIC/SASIC profiler JSON files and print key metrics side-by-side.

Usage:
    python scripts/compare_stats.py baseline_stats.json sasic_stats.json
"""

import json
import argparse
import sys
from typing import Dict, Any, Optional


def safe_get(d: Dict[str, Any], *keys, default=None):
    """Safely get nested dict value."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, {})
    return d if isinstance(d, (dict, list)) or d is not None else default


def format_time(seconds: Optional[float]) -> str:
    """Format seconds as human-readable time."""
    if seconds is None:
        return "N/A"
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds:.2f}s"


def format_number(val: Optional[float]) -> str:
    """Format number with commas."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:,.2f}" if val >= 1 else f"{val:.4f}"
    return f"{val:,}"


def compare_phases(stats1: Dict, stats2: Dict, name1: str, name2: str):
    """Compare phase timings."""
    print("\n" + "="*80)
    print("PHASE TIMINGS")
    print("="*80)
    print(f"{'Phase':<20} {name1:>20} {name2:>20} {'Delta':>15}")
    print("-"*80)
    
    phases = ["filtering", "clustering", "merging", "verification"]
    for phase in phases:
        t1 = safe_get(stats1, "phases", phase, "total_seconds")
        t2 = safe_get(stats2, "phases", phase, "total_seconds")
        delta = (t2 - t1) if (t1 is not None and t2 is not None) else None
        delta_str = format_time(delta) if delta is not None else "N/A"
        print(f"{phase:<20} {format_time(t1):>20} {format_time(t2):>20} {delta_str:>15}")


def compare_k_trials(stats1: Dict, stats2: Dict, name1: str, name2: str):
    """Compare k-trials statistics."""
    sic1 = safe_get(stats1, "sic", default={})
    sic2 = safe_get(stats2, "sic", default={})
    
    if not sic1 and not sic2:
        print("\n" + "="*80)
        print("K-TRIALS STATISTICS")
        print("="*80)
        print("(Not available - track_k_trials was not enabled)")
        return
    
    print("\n" + "="*80)
    print("K-TRIALS STATISTICS")
    print("="*80)
    
    # Global stats
    total1 = sic1.get("total_k_trials")
    total2 = sic2.get("total_k_trials")
    avg1 = sic1.get("avg_k_trials_per_neuron")
    avg2 = sic2.get("avg_k_trials_per_neuron")
    num_neurons1 = sic1.get("num_neurons_evaluated")
    num_neurons2 = sic2.get("num_neurons_evaluated")
    
    print(f"\n{'Metric':<40} {name1:>20} {name2:>20} {'Delta':>15}")
    print("-"*80)
    
    if total1 is not None or total2 is not None:
        delta = (total2 - total1) if (total1 is not None and total2 is not None) else None
        delta_str = format_number(delta) if delta is not None else "N/A"
        print(f"{'Total k-trials':<40} {format_number(total1):>20} {format_number(total2):>20} {delta_str:>15}")
    
    if num_neurons1 is not None or num_neurons2 is not None:
        delta = (num_neurons2 - num_neurons1) if (num_neurons1 is not None and num_neurons2 is not None) else None
        delta_str = format_number(delta) if delta is not None else "N/A"
        print(f"{'Neurons evaluated':<40} {format_number(num_neurons1):>20} {format_number(num_neurons2):>20} {delta_str:>15}")
    
    if avg1 is not None or avg2 is not None:
        delta = (avg2 - avg1) if (avg1 is not None and avg2 is not None) else None
        delta_str = format_number(delta) if delta is not None else "N/A"
        print(f"{'Avg k-trials per neuron':<40} {format_number(avg1):>20} {format_number(avg2):>20} {delta_str:>15}")
    
    # Per-layer stats
    layers1 = safe_get(stats1, "layers", default={})
    layers2 = safe_get(stats2, "layers", default={})
    
    all_layer_names = set(layers1.keys()) | set(layers2.keys())
    if all_layer_names:
        print(f"\n{'Layer':<30} {'Metric':<15} {name1:>15} {name2:>15} {'Delta':>15}")
        print("-"*80)
        
        for layer_name in sorted(all_layer_names):
            l1 = layers1.get(layer_name, {})
            l2 = layers2.get(layer_name, {})
            
            avg1 = l1.get("avg_k_trials_per_neuron")
            avg2 = l2.get("avg_k_trials_per_neuron")
            total1 = l1.get("total_k_trials")
            total2 = l2.get("total_k_trials")
            
            if avg1 is not None or avg2 is not None:
                delta = (avg2 - avg1) if (avg1 is not None and avg2 is not None) else None
                delta_str = format_number(delta) if delta is not None else "N/A"
                print(f"{layer_name:<30} {'avg_k_trials':<15} {format_number(avg1):>15} {format_number(avg2):>15} {delta_str:>15}")
            
            if total1 is not None or total2 is not None:
                delta = (total2 - total1) if (total1 is not None and total2 is not None) else None
                delta_str = format_number(delta) if delta is not None else "N/A"
                print(f"{layer_name:<30} {'total_k_trials':<15} {format_number(total1):>15} {format_number(total2):>15} {delta_str:>15}")


def compare_global_stats(stats1: Dict, stats2: Dict, name1: str, name2: str):
    """Compare global statistics."""
    global1 = safe_get(stats1, "global", default={})
    global2 = safe_get(stats2, "global", default={})
    
    if not global1 and not global2:
        return
    
    print("\n" + "="*80)
    print("GLOBAL STATISTICS")
    print("="*80)
    print(f"{'Metric':<40} {name1:>20} {name2:>20} {'Delta':>15}")
    print("-"*80)
    
    metrics = [
        ("original_params", "Original params"),
        ("final_params", "Final params"),
        ("effective_params", "Effective params"),
        ("zero_params", "Zero params"),
        ("compression_ratio", "Compression ratio"),
    ]
    
    for key, label in metrics:
        val1 = global1.get(key)
        val2 = global2.get(key)
        if val1 is not None or val2 is not None:
            delta = (val2 - val1) if (val1 is not None and val2 is not None) else None
            delta_str = format_number(delta) if delta is not None else "N/A"
            print(f"{label:<40} {format_number(val1):>20} {format_number(val2):>20} {delta_str:>15}")


def compare_accuracy(stats1: Dict, stats2: Dict, name1: str, name2: str):
    """Compare accuracy if available."""
    # Accuracy might be in different places - check common locations
    acc1 = safe_get(stats1, "final_accuracy") or safe_get(stats1, "accuracy", "final")
    acc2 = safe_get(stats2, "final_accuracy") or safe_get(stats2, "accuracy", "final")
    
    if acc1 is None and acc2 is None:
        return
    
    print("\n" + "="*80)
    print("ACCURACY")
    print("="*80)
    print(f"{'Metric':<40} {name1:>20} {name2:>20} {'Delta':>15}")
    print("-"*80)
    
    if acc1 is not None or acc2 is not None:
        delta = (acc2 - acc1) if (acc1 is not None and acc2 is not None) else None
        delta_str = f"{delta:+.2f}%" if delta is not None else "N/A"
        acc1_str = f"{acc1:.2f}%" if acc1 is not None else "N/A"
        acc2_str = f"{acc2:.2f}%" if acc2 is not None else "N/A"
        print(f"{'Final accuracy':<40} {acc1_str:>20} {acc2_str:>20} {delta_str:>15}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two SIC/SASIC profiler JSON files"
    )
    parser.add_argument(
        "file1",
        type=str,
        help="First JSON stats file (e.g., baseline)"
    )
    parser.add_argument(
        "file2",
        type=str,
        help="Second JSON stats file (e.g., SASIC)"
    )
    parser.add_argument(
        "--name1",
        type=str,
        default=None,
        help="Label for first file (default: filename)"
    )
    parser.add_argument(
        "--name2",
        type=str,
        default=None,
        help="Label for second file (default: filename)"
    )
    
    args = parser.parse_args()
    
    # Load JSON files
    try:
        with open(args.file1, "r") as f:
            stats1 = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.file1}: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        with open(args.file2, "r") as f:
            stats2 = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.file2}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine names
    name1 = args.name1 or args.file1.split("/")[-1].replace("_stats.json", "")
    name2 = args.name2 or args.file2.split("/")[-1].replace("_stats.json", "")
    
    # Print header
    print("="*80)
    print(f"COMPARISON: {name1} vs {name2}")
    print("="*80)
    
    # Compare sections
    compare_phases(stats1, stats2, name1, name2)
    compare_k_trials(stats1, stats2, name1, name2)
    compare_global_stats(stats1, stats2, name1, name2)
    compare_accuracy(stats1, stats2, name1, name2)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
