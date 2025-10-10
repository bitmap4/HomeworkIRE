#!/usr/bin/env python3
"""Generate plots from existing metrics JSON file."""

import json
from pathlib import Path
import hydra
from omegaconf import DictConfig

from indexing_and_retrieval.utils.metrics import PerformanceMetrics

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    metrics_dir = Path(cfg.paths.metrics_dir)
    plots_dir = Path(cfg.paths.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics
    metrics_file = metrics_dir / "all_metrics.json"
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found!")
        return
    
    print(f"Loading metrics from {metrics_file}...")
    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)
    
    all_latency_metrics = all_metrics['latency']
    all_throughput_metrics = all_metrics['throughput']
    all_memory_metrics = all_metrics['memory']
    all_functional_metrics = all_metrics['functional']
    
    print("Generating plots...")
    perf_metrics = PerformanceMetrics(metrics_dir)
    
    # Generate all comparison plots
    perf_metrics.plot_by_info_type(all_latency_metrics, all_throughput_metrics, all_memory_metrics, 
                                   all_functional_metrics, 'plot_c_comparison_by_info_type.png')
    
    perf_metrics.plot_by_datastore(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                   all_functional_metrics, 'plot_a_comparison_by_datastore.png')
    
    perf_metrics.plot_by_compression(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                     all_functional_metrics, 'plot_ab_comparison_by_compression.png')
    
    perf_metrics.plot_by_skip_pointers(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                       all_functional_metrics, 'plot_a_comparison_skip_pointers.png')
    
    perf_metrics.plot_by_query_processing(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                          all_functional_metrics, 'plot_ac_comparison_query_processing.png')
    
    print(f"✓ Plots saved to: {plots_dir.absolute()}")

if __name__ == "__main__":
    main()
