#!/usr/bin/env python3
"""
Unit tests for summarize_sic.py

Tests that the summarizer:
- Runs without crashing
- Produces summaries containing key fields
- Detects anomalies correctly
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from summarize_sic import (
    load_json,
    generate_console_summary,
    generate_markdown_summary,
    check_weirdness_flags,
    analyze_failure_reasons,
)


def create_minimal_fixture() -> dict:
    """Create a minimal valid SASIC JSON fixture for testing."""
    return {
        "global": {
            "start_time": "2025-12-12T21:59:38.638186",
            "end_time": "2025-12-12T21:59:49.220694",
            "total_duration": 10.582503813318908,
            "original_params": 101770,
            "final_params": 101770,
            "effective_params": 12076,
            "zero_params": 66640,
            "compression_ratio": 1.0,
            "memory_peak_mb": 784.5703125,
            "memory_saved_mb": -112.59765625
        },
        "phases": {
            "initialization": {
                "duration": 4.99221496284008e-05,
                "start_time": 3734873.315746889
            },
            "filtering": {
                "duration": 0.16726900869980454,
                "samples_before": None,
                "samples_after": 32,
                "start_time": 3734873.31585018
            },
            "clustering": {
                "duration": 0.8717855582945049,
                "total_neurons": 138,
                "successful_neurons": 117,
                "start_time": 3734882.133551005
            },
            "merging": {
                "duration": 0.8908669287338853,
                "merges_performed": 67,
                "start_time": 3734883.005482485
            },
            "verification": {
                "duration": 5.333404988050461e-05,
                "checks_performed": 0,
                "start_time": 3734883.897801476
            }
        },
        "layers": {
            "fc1": {
                "shape": [128, 784],
                "total_params": 100352,
                "neurons_processed": 128,
                "neurons_successful": 114,
                "neurons_failed": 14,
                "original_unique_weights": 100300,
                "final_unique_weights": 100300,
                "zero_weights_before": 0,
                "zero_weights_after": 0,
                "clustering_attempts": [
                    {
                        "layer": "fc1",
                        "neuron": 0,
                        "clusters": 0,
                        "success": True,
                        "timestamp": 8.826490317005664,
                        "reason": "zeroed_perceptron"
                    },
                    {
                        "layer": "fc1",
                        "neuron": 24,
                        "clusters": 0,
                        "success": False,
                        "timestamp": 8.891927327029407,
                        "reason": "zero_rejected"
                    },
                    {
                        "layer": "fc1",
                        "neuron": 24,
                        "clusters": 1,
                        "success": False,
                        "timestamp": 8.89791440218687,
                        "reason": "accuracy_loss"
                    },
                    {
                        "layer": "fc1",
                        "neuron": 24,
                        "clusters": 2,
                        "success": True,
                        "timestamp": 8.90277708740905,
                        "reason": "converged"
                    },
                    {
                        "layer": "fc1",
                        "neuron": 25,
                        "clusters": 3,
                        "success": False,
                        "timestamp": 8.919754800386727,
                        "reason": "max_clusters_exceeded"
                    }
                ],
                "processing_time": 0.7912946566939354,
                "memory_usage_mb": 0.0,
                "weight_distribution": {
                    "before": {
                        "min": -0.7830150127410889,
                        "max": 0.6722612380981445,
                        "mean": -3.654391912277788e-05,
                        "std": 0.11156047880649567
                    },
                    "after": {
                        "min": -0.7830150127410889,
                        "max": 0.6722612380981445,
                        "mean": -3.654391912277788e-05,
                        "std": 0.11156047880649567
                    }
                },
                "k_trials_list": [0, 0, 2, 3, 0],
                "avg_k_trials_per_neuron": 0.8046875,
                "total_k_trials": 103,
                "success_rate": 0.890625
            },
            "fc2": {
                "shape": [10, 128],
                "total_params": 1280,
                "neurons_processed": 10,
                "neurons_successful": 3,
                "neurons_failed": 7,
                "original_unique_weights": 1280,
                "final_unique_weights": 1280,
                "zero_weights_before": 0,
                "zero_weights_after": 0,
                "clustering_attempts": [
                    {
                        "layer": "fc2",
                        "neuron": 0,
                        "clusters": 3,
                        "success": True,
                        "timestamp": 9.616997173987329,
                        "reason": "converged"
                    },
                    {
                        "layer": "fc2",
                        "neuron": 1,
                        "clusters": 3,
                        "success": False,
                        "timestamp": 9.623703110963106,
                        "reason": "max_clusters_exceeded"
                    }
                ],
                "processing_time": 0.062023378908634186,
                "memory_usage_mb": 0.0,
                "weight_distribution": {
                    "before": {
                        "min": -0.8796645402908325,
                        "max": 0.626733660697937,
                        "mean": -0.03128276392817497,
                        "std": 0.23860426247119904
                    },
                    "after": {
                        "min": -0.8796645402908325,
                        "max": 0.626733660697937,
                        "mean": -0.03128276392817497,
                        "std": 0.23860426247119904
                    }
                },
                "k_trials_list": [3, 3],
                "avg_k_trials_per_neuron": 2.7,
                "total_k_trials": 27,
                "success_rate": 0.3
            }
        },
        "convergence": {
            "clustering_attempts": []
        }
    }


class TestSASICSummary(unittest.TestCase):
    """Test cases for SASIC summary generator."""
    
    def setUp(self):
        """Set up test fixture."""
        self.fixture_data = create_minimal_fixture()
    
    def test_load_json(self):
        """Test JSON loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.fixture_data, f)
            temp_path = Path(f.name)
        
        try:
            loaded = load_json(temp_path)
            self.assertEqual(loaded["global"]["total_duration"], 10.582503813318908)
            self.assertEqual(loaded["layers"]["fc1"]["neurons_processed"], 128)
        finally:
            temp_path.unlink()
    
    def test_console_summary_contains_key_fields(self):
        """Test that console summary contains expected key fields."""
        summary = generate_console_summary(self.fixture_data)
        
        # Check for key fields
        self.assertIn("total duration", summary.lower())
        self.assertIn("fc1", summary)
        self.assertIn("success rate", summary.lower())
        self.assertIn("zero_rejected", summary.lower())
        self.assertIn("compression ratio", summary.lower())
        self.assertIn("original params", summary.lower())
        self.assertIn("final params", summary.lower())
    
    def test_markdown_summary_contains_key_fields(self):
        """Test that markdown summary contains expected key fields."""
        summary = generate_markdown_summary(self.fixture_data)
        
        # Check for key fields
        self.assertIn("total duration", summary.lower())
        self.assertIn("fc1", summary)
        self.assertIn("success rate", summary.lower())
        self.assertIn("zero_rejected", summary.lower())
        self.assertIn("#", summary)  # Markdown headers
    
    def test_failure_reasons_analysis(self):
        """Test failure reason aggregation."""
        reasons = analyze_failure_reasons(self.fixture_data["layers"])
        
        self.assertIn("zero_rejected", reasons)
        self.assertIn("accuracy_loss", reasons)
        self.assertIn("converged", reasons)
        self.assertIn("max_clusters_exceeded", reasons)
        self.assertGreater(reasons["zero_rejected"], 0)
    
    def test_weirdness_flags_detection(self):
        """Test anomaly detection."""
        warnings = check_weirdness_flags(self.fixture_data)
        
        # Should detect: no compression, negative memory saved, no verification
        flag_names = [flag[0] for flag in warnings]
        self.assertIn("no_compression", flag_names)
        self.assertIn("negative_memory_saved", flag_names)
        self.assertIn("no_verification", flag_names)
    
    def test_summary_runs_without_crashing(self):
        """Test that summary generation doesn't crash."""
        try:
            console = generate_console_summary(self.fixture_data)
            markdown = generate_markdown_summary(self.fixture_data)
            self.assertIsInstance(console, str)
            self.assertIsInstance(markdown, str)
            self.assertGreater(len(console), 0)
            self.assertGreater(len(markdown), 0)
        except Exception as e:
            self.fail(f"Summary generation raised exception: {e}")
    
    def test_high_zeroed_fraction_detection(self):
        """Test detection of high fraction of zeroed perceptrons."""
        # Create data with >50% zeroed
        data = create_minimal_fixture()
        # Add more zeroed_perceptron attempts
        for i in range(100):
            data["layers"]["fc1"]["clustering_attempts"].append({
                "layer": "fc1",
                "neuron": i + 200,
                "clusters": 0,
                "success": True,
                "timestamp": 10.0 + i * 0.01,
                "reason": "zeroed_perceptron"
            })
        
        warnings = check_weirdness_flags(data)
        flag_names = [flag[0] for flag in warnings]
        self.assertIn("high_zeroed_fraction", flag_names)


if __name__ == "__main__":
    unittest.main()

