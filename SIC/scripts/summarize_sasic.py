#!/usr/bin/env python3
"""
SASIC Run Summary Generator

This script analyzes SASIC run JSON files and generates human-readable summaries
with anomaly detection and derived metrics.

JSON Schema Assumptions:
- Top-level keys: "global", "phases", "layers", "convergence"
- "global": contains timing, parameter counts, compression metrics, memory stats
- "phases": contains duration and metadata for initialization, filtering, clustering, merging, verification
- "layers": dict mapping layer names to layer stats including:
  - shape, total_params, neurons_processed/successful/failed
  - original_unique_weights, final_unique_weights
  - zero_weights_before, zero_weights_after
  - clustering_attempts (list of attempt dicts with layer, neuron, clusters, success, timestamp, reason)
  - weight_distribution (before/after with min/max/mean/std)
  - k_trials_list, avg_k_trials_per_neuron, total_k_trials, success_rate
- "convergence": contains aggregated clustering_attempts (optional, may duplicate layer data)
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple
from statistics import mean


def load_json(json_path: Path) -> Dict[str, Any]:
    """Load and parse JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"


def calculate_phase_percentages(phases: Dict[str, Any], total_duration: float) -> Dict[str, float]:
    """Calculate percentage of total duration for each phase."""
    percentages = {}
    for phase_name, phase_data in phases.items():
        if isinstance(phase_data, dict) and "duration" in phase_data:
            pct = (phase_data["duration"] / total_duration * 100) if total_duration > 0 else 0.0
            percentages[phase_name] = pct
    return percentages


def analyze_failure_reasons(layers: Dict[str, Any]) -> Counter:
    """Aggregate failure reasons across all clustering attempts."""
    reason_counter = Counter()
    for layer_name, layer_data in layers.items():
        if isinstance(layer_data, dict) and "clustering_attempts" in layer_data:
            for attempt in layer_data["clustering_attempts"]:
                if isinstance(attempt, dict) and "reason" in attempt:
                    reason_counter[attempt["reason"]] += 1
    return reason_counter


def check_weirdness_flags(data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Check for anomalies and return list of (flag_name, warning_message) tuples.
    """
    warnings = []
    global_data = data.get("global", {})
    layers = data.get("layers", {})
    phases = data.get("phases", {})
    
    # Check 1: No compression occurred
    compression_ratio = global_data.get("compression_ratio", 1.0)
    all_layers_no_compression = True
    for layer_name, layer_data in layers.items():
        if isinstance(layer_data, dict):
            orig_unique = layer_data.get("original_unique_weights")
            final_unique = layer_data.get("final_unique_weights")
            if orig_unique is not None and final_unique is not None:
                if orig_unique != final_unique:
                    all_layers_no_compression = False
                    break
    
    if compression_ratio == 1.0 and all_layers_no_compression:
        warnings.append((
            "no_compression",
            "No compression/weight sharing occurred; check whether consolidation is being applied or only logged."
        ))
    
    # Check 2: Negative memory saved
    memory_saved = global_data.get("memory_saved_mb", 0)
    if memory_saved < 0:
        warnings.append((
            "negative_memory_saved",
            f"Negative memory_saved_mb ({memory_saved:.2f} MB) implies overhead increased memory (e.g., buffers/copies); verify measurement method."
        ))
    
    # Check 3: No verification checks
    verification = phases.get("verification", {})
    checks_performed = verification.get("checks_performed", 0)
    if checks_performed == 0:
        warnings.append((
            "no_verification",
            "No verification checks ran; results may be unvalidated."
        ))
    
    # Check 4: Large fraction of zeroed perceptrons
    total_neurons = 0
    zeroed_neurons = 0
    for layer_name, layer_data in layers.items():
        if isinstance(layer_data, dict) and "clustering_attempts" in layer_data:
            neuron_zeroed = set()
            for attempt in layer_data["clustering_attempts"]:
                if isinstance(attempt, dict):
                    neuron_id = attempt.get("neuron")
                    if neuron_id is not None:
                        if neuron_id not in neuron_zeroed:
                            neuron_zeroed.add(neuron_id)
                            total_neurons += 1
                        if attempt.get("reason") == "zeroed_perceptron" and attempt.get("success"):
                            zeroed_neurons += 1
    
    if total_neurons > 0:
        zeroed_fraction = zeroed_neurons / total_neurons
        if zeroed_fraction > 0.5:  # More than 50% zeroed
            warnings.append((
                "high_zeroed_fraction",
                f"Large fraction of neurons were zeroed ({zeroed_fraction:.1%}); ensure this is intended and consistent with acceptance rules."
            ))
    
    # Check 5: Duplicate terminal attempts (same neuron, same K, both logged)
    for layer_name, layer_data in layers.items():
        if isinstance(layer_data, dict) and "clustering_attempts" in layer_data:
            neuron_attempts = defaultdict(list)
            for attempt in layer_data["clustering_attempts"]:
                if isinstance(attempt, dict):
                    neuron_id = attempt.get("neuron")
                    clusters = attempt.get("clusters")
                    reason = attempt.get("reason")
                    if neuron_id is not None and clusters is not None:
                        neuron_attempts[neuron_id].append((clusters, reason))
            
            for neuron_id, attempts in neuron_attempts.items():
                # Check for same K with both a regular attempt and max_clusters_exceeded
                k_to_reasons = defaultdict(set)
                for k, reason in attempts:
                    k_to_reasons[k].add(reason)
                
                for k, reasons in k_to_reasons.items():
                    # If max_clusters_exceeded is present along with other reasons at same K
                    if "max_clusters_exceeded" in reasons and len(reasons) > 1:
                        warnings.append((
                            "duplicate_terminal_attempt",
                            f"Neuron {neuron_id} in layer {layer_name} has duplicate terminal attempt at K={k} (both regular attempt and max_clusters_exceeded logged); check loop/termination logging."
                        ))
                        break  # Only warn once per neuron
    
    return warnings


def generate_console_summary(data: Dict[str, Any]) -> str:
    """Generate concise console summary."""
    lines = []
    global_data = data.get("global", {})
    phases = data.get("phases", {})
    layers = data.get("layers", {})
    
    # Header
    lines.append("=" * 80)
    lines.append("SASIC Run Summary")
    lines.append("=" * 80)
    
    # Timing
    total_duration = global_data.get("total_duration", 0)
    start_time = global_data.get("start_time", "N/A")
    end_time = global_data.get("end_time", "N/A")
    lines.append(f"\nTiming:")
    lines.append(f"  Start: {start_time}")
    lines.append(f"  End:   {end_time}")
    lines.append(f"  Total Duration: {format_duration(total_duration)}")
    
    # Phase breakdown
    phase_pcts = calculate_phase_percentages(phases, total_duration)
    lines.append(f"\nPhase Breakdown:")
    for phase_name in ["initialization", "filtering", "clustering", "merging", "verification"]:
        if phase_name in phases:
            phase_data = phases[phase_name]
            duration = phase_data.get("duration", 0)
            pct = phase_pcts.get(phase_name, 0)
            lines.append(f"  {phase_name.capitalize():15s}: {format_duration(duration):>10s} ({pct:5.1f}%)")
    
    # Global metrics
    lines.append(f"\nGlobal Metrics:")
    lines.append(f"  Original Params:     {global_data.get('original_params', 0):,}")
    lines.append(f"  Final Params:         {global_data.get('final_params', 0):,}")
    lines.append(f"  Effective Params:    {global_data.get('effective_params', 0):,}")
    lines.append(f"  Zero Params:         {global_data.get('zero_params', 0):,}")
    lines.append(f"  Compression Ratio:   {global_data.get('compression_ratio', 1.0):.4f}")
    lines.append(f"  Memory Peak:         {global_data.get('memory_peak_mb', 0):.2f} MB")
    lines.append(f"  Memory Saved:        {global_data.get('memory_saved_mb', 0):.2f} MB")
    
    # Per-layer summary
    lines.append(f"\nLayer Summary:")
    for layer_name in sorted(layers.keys()):
        layer_data = layers[layer_name]
        if not isinstance(layer_data, dict):
            continue
        
        shape = layer_data.get("shape", [])
        total_params = layer_data.get("total_params", 0)
        neurons_processed = layer_data.get("neurons_processed", 0)
        neurons_successful = layer_data.get("neurons_successful", 0)
        neurons_failed = layer_data.get("neurons_failed", 0)
        success_rate = layer_data.get("success_rate", 0.0)
        total_k_trials = layer_data.get("total_k_trials", 0)
        avg_k_trials = layer_data.get("avg_k_trials_per_neuron", 0.0)
        orig_unique = layer_data.get("original_unique_weights", 0)
        final_unique = layer_data.get("final_unique_weights", 0)
        zero_before = layer_data.get("zero_weights_before", 0)
        zero_after = layer_data.get("zero_weights_after", 0)
        
        lines.append(f"\n  {layer_name}:")
        lines.append(f"    Shape:                    {shape}")
        lines.append(f"    Total Params:             {total_params:,}")
        lines.append(f"    Neurons:                  {neurons_processed} processed, {neurons_successful} successful, {neurons_failed} failed")
        lines.append(f"    Success Rate:             {success_rate:.1%}")
        lines.append(f"    K Trials:                 {total_k_trials} total, {avg_k_trials:.2f} avg/neuron")
        lines.append(f"    Unique Weights:           {orig_unique:,} → {final_unique:,}")
        lines.append(f"    Zero Weights:             {zero_before:,} → {zero_after:,}")
        
        # Weight distribution
        wdist = layer_data.get("weight_distribution", {})
        if isinstance(wdist, dict):
            before = wdist.get("before", {})
            after = wdist.get("after", {})
            if before and after:
                lines.append(f"    Weight Dist (before):     min={before.get('min', 0):.4f}, max={before.get('max', 0):.4f}, mean={before.get('mean', 0):.4f}, std={before.get('std', 0):.4f}")
                lines.append(f"    Weight Dist (after):      min={after.get('min', 0):.4f}, max={after.get('max', 0):.4f}, mean={after.get('mean', 0):.4f}, std={after.get('std', 0):.4f}")
    
    # Failure reasons
    failure_reasons = analyze_failure_reasons(layers)
    if failure_reasons:
        lines.append(f"\nTop Failure Reasons (across all attempts):")
        for reason, count in failure_reasons.most_common(10):
            lines.append(f"  {reason:30s}: {count:,}")
    
    # Weirdness flags
    warnings = check_weirdness_flags(data)
    if warnings:
        lines.append(f"\n⚠️  Anomalies/Warnings:")
        for flag_name, message in warnings:
            lines.append(f"  [{flag_name}] {message}")
    else:
        lines.append(f"\n✓ No anomalies detected")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def generate_markdown_summary(data: Dict[str, Any]) -> str:
    """Generate detailed Markdown summary."""
    lines = []
    global_data = data.get("global", {})
    phases = data.get("phases", {})
    layers = data.get("layers", {})
    
    # Header
    lines.append("# SASIC Run Summary")
    lines.append("")
    lines.append(f"**Generated:** {global_data.get('start_time', 'N/A')}")
    lines.append("")
    
    # Timing
    total_duration = global_data.get("total_duration", 0)
    lines.append("## Timing")
    lines.append("")
    lines.append(f"- **Start Time:** {global_data.get('start_time', 'N/A')}")
    lines.append(f"- **End Time:** {global_data.get('end_time', 'N/A')}")
    lines.append(f"- **Total Duration:** {format_duration(total_duration)}")
    lines.append("")
    
    # Phase breakdown
    phase_pcts = calculate_phase_percentages(phases, total_duration)
    lines.append("## Phase Breakdown")
    lines.append("")
    lines.append("| Phase | Duration | Percentage |")
    lines.append("|-------|----------|------------|")
    for phase_name in ["initialization", "filtering", "clustering", "merging", "verification"]:
        if phase_name in phases:
            phase_data = phases[phase_name]
            duration = phase_data.get("duration", 0)
            pct = phase_pcts.get(phase_name, 0)
            lines.append(f"| {phase_name.capitalize()} | {format_duration(duration)} | {pct:.1f}% |")
    lines.append("")
    
    # Global metrics
    lines.append("## Global Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Original Params | {global_data.get('original_params', 0):,} |")
    lines.append(f"| Final Params | {global_data.get('final_params', 0):,} |")
    lines.append(f"| Effective Params | {global_data.get('effective_params', 0):,} |")
    lines.append(f"| Zero Params | {global_data.get('zero_params', 0):,} |")
    lines.append(f"| Compression Ratio | {global_data.get('compression_ratio', 1.0):.4f} |")
    lines.append(f"| Memory Peak (MB) | {global_data.get('memory_peak_mb', 0):.2f} |")
    lines.append(f"| Memory Saved (MB) | {global_data.get('memory_saved_mb', 0):.2f} |")
    lines.append("")
    
    # Per-layer details
    lines.append("## Layer Details")
    lines.append("")
    for layer_name in sorted(layers.keys()):
        layer_data = layers[layer_name]
        if not isinstance(layer_data, dict):
            continue
        
        lines.append(f"### {layer_name}")
        lines.append("")
        
        shape = layer_data.get("shape", [])
        total_params = layer_data.get("total_params", 0)
        neurons_processed = layer_data.get("neurons_processed", 0)
        neurons_successful = layer_data.get("neurons_successful", 0)
        neurons_failed = layer_data.get("neurons_failed", 0)
        success_rate = layer_data.get("success_rate", 0.0)
        total_k_trials = layer_data.get("total_k_trials", 0)
        avg_k_trials = layer_data.get("avg_k_trials_per_neuron", 0.0)
        orig_unique = layer_data.get("original_unique_weights", 0)
        final_unique = layer_data.get("final_unique_weights", 0)
        zero_before = layer_data.get("zero_weights_before", 0)
        zero_after = layer_data.get("zero_weights_after", 0)
        
        lines.append("#### Basic Stats")
        lines.append("")
        lines.append(f"- **Shape:** {shape}")
        lines.append(f"- **Total Params:** {total_params:,}")
        lines.append(f"- **Neurons Processed:** {neurons_processed}")
        lines.append(f"- **Neurons Successful:** {neurons_successful}")
        lines.append(f"- **Neurons Failed:** {neurons_failed}")
        lines.append(f"- **Success Rate:** {success_rate:.1%}")
        lines.append("")
        
        lines.append("#### Clustering Stats")
        lines.append("")
        lines.append(f"- **Total K Trials:** {total_k_trials}")
        lines.append(f"- **Avg K Trials per Neuron:** {avg_k_trials:.2f}")
        lines.append("")
        
        lines.append("#### Weight Stats")
        lines.append("")
        lines.append(f"- **Original Unique Weights:** {orig_unique:,}")
        lines.append(f"- **Final Unique Weights:** {final_unique:,}")
        lines.append(f"- **Zero Weights (before):** {zero_before:,}")
        lines.append(f"- **Zero Weights (after):** {zero_after:,}")
        lines.append("")
        
        # Weight distribution
        wdist = layer_data.get("weight_distribution", {})
        if isinstance(wdist, dict):
            before = wdist.get("before", {})
            after = wdist.get("after", {})
            if before and after:
                lines.append("#### Weight Distribution")
                lines.append("")
                lines.append("| Stat | Before | After |")
                lines.append("|------|--------|-------|")
                lines.append(f"| Min | {before.get('min', 0):.4f} | {after.get('min', 0):.4f} |")
                lines.append(f"| Max | {before.get('max', 0):.4f} | {after.get('max', 0):.4f} |")
                lines.append(f"| Mean | {before.get('mean', 0):.4f} | {after.get('mean', 0):.4f} |")
                lines.append(f"| Std | {before.get('std', 0):.4f} | {after.get('std', 0):.4f} |")
                lines.append("")
    
    # Failure reasons
    failure_reasons = analyze_failure_reasons(layers)
    if failure_reasons:
        lines.append("## Failure Reasons")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("|--------|-------|")
        for reason, count in failure_reasons.most_common(20):
            lines.append(f"| {reason} | {count:,} |")
        lines.append("")
    
    # Weirdness flags
    warnings = check_weirdness_flags(data)
    if warnings:
        lines.append("## ⚠️ Anomalies and Warnings")
        lines.append("")
        for flag_name, message in warnings:
            lines.append(f"### [{flag_name}]")
            lines.append("")
            lines.append(f"{message}")
            lines.append("")
    else:
        lines.append("## ✓ Status")
        lines.append("")
        lines.append("No anomalies detected.")
        lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate human-readable summary from SASIC run JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to SASIC results JSON file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write Markdown summary (default: print only)",
    )
    
    args = parser.parse_args()
    
    # Load JSON
    if not args.json.exists():
        print(f"Error: JSON file not found: {args.json}", file=sys.stderr)
        return 1
    
    data = load_json(args.json)
    
    # Generate summaries
    console_summary = generate_console_summary(data)
    print(console_summary)
    
    if args.out:
        markdown_summary = generate_markdown_summary(data)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(markdown_summary)
        print(f"\nMarkdown summary written to: {args.out}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

