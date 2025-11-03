import time
import psutil
import numpy as np
from typing import List, Dict, Callable, Set
from collections import defaultdict
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceMetrics:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_latency(self, query_func: Callable, queries: List[str], 
                       num_warmup: int = 5, num_iterations: int = 100) -> Dict:
        print(f"Measuring latency with {num_warmup} warmup and {num_iterations} iterations...")
        
        for query in queries[:num_warmup]:
            query_func(query)
        
        latencies = []
        for _ in range(num_iterations):
            for query in queries:
                start_time = time.perf_counter()
                query_func(query)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)
        
        latencies_sorted = sorted(latencies)
        
        return {
            'mean': np.mean(latencies),
            'median': np.median(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'total_queries': len(latencies)
        }
    
    def measure_throughput(self, query_func: Callable, queries: List[str], 
                          duration_seconds: int = 10) -> Dict:
        print(f"Measuring throughput for {duration_seconds} seconds...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        query_count = 0
        query_idx = 0
        max_queries = 50  # Safety limit to prevent infinite loops
        
        while time.time() < end_time and query_count < max_queries:
            query = queries[query_idx % len(queries)]
            try:
                query_func(query)
                query_count += 1
                query_idx += 1
            except Exception as e:
                print(f"Warning: Query failed during throughput test: {e}")
                break
        
        actual_duration = time.time() - start_time
        throughput = query_count / actual_duration if actual_duration > 0 else 0
        
        return {
            'queries_per_second': throughput,
            'total_queries': query_count,
            'duration_seconds': actual_duration
        }
    
    def measure_memory_footprint(self, index_obj=None) -> Dict:
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent': process.memory_percent()
        }
    
    def compute_precision_recall(self, retrieved: Set[str], relevant: Set[str]) -> Dict:
        if not retrieved:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        true_positives = len(retrieved & relevant)
        
        precision = true_positives / len(retrieved) if retrieved else 0.0
        recall = true_positives / len(relevant) if relevant else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'retrieved_count': len(retrieved),
            'relevant_count': len(relevant)
        }
    
    def save_metrics(self, metrics: Dict, filename: str):
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
    
    def plot_by_info_type(self, metrics_data: Dict[str, Dict], metric_name: str, filename: str):
        """Plot.C for x=n: Compare different information types (BOOLEAN, WORDCOUNT, TFIDF)"""
        # Extract variants that differ only in info type
        variants = {}
        for key, value in metrics_data.items():
            if key.startswith('SelfIndex-v1.'):
                info_type = key.split('.')[1][0]  # Extract x from v1.xyziq
                variants[info_type] = value
        
        if not variants:
            print(f"Warning: No variants found for info type comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        info_labels = {'1': 'Boolean', '2': 'WordCount', '3': 'TF-IDF'}
        
        sorted_keys = sorted(variants.keys())
        x_pos = np.arange(len(sorted_keys))
        values = [variants[k][metric_name] for k in sorted_keys]
        
        bars = ax.bar(x_pos, values, alpha=0.8, color='steelblue')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Information Type (x)', fontsize=12)
        ax.set_ylabel(f'{metric_name.capitalize()} (MB)', fontsize=12)
        ax.set_title(f'Plot.C: Memory Footprint by Information Type', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([info_labels[k] for k in sorted_keys])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
    
    def plot_by_datastore(self, metrics_data: Dict[str, Dict], metric_name: str, filename: str):
        """Plot.A for y=n: Compare datastore choices (CUSTOM, DB1, DB2)"""
        variants = {}
        for key, value in metrics_data.items():
            if key.startswith('SelfIndex-v1.'):
                dstore_type = key.split('.')[1][1]  # Extract y from v1.xyziq
                variants[dstore_type] = value
        
        if not variants:
            print(f"Warning: No variants found for datastore comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        dstore_labels = {'1': 'Custom Disk', '2': 'PostgreSQL', '3': 'Redis'}
        
        sorted_keys = sorted(variants.keys())
        x_pos = np.arange(len(sorted_keys))
        values = [variants[k][metric_name] for k in sorted_keys]
        
        bars = ax.bar(x_pos, values, alpha=0.8, color='coral')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Datastore Type (y)', fontsize=12)
        ax.set_ylabel(f'{metric_name.upper() if metric_name in ["p95", "p99"] else metric_name.capitalize()} (ms)', fontsize=12)
        ax.set_title(f'Plot.A: Latency by Datastore Type', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([dstore_labels[k] for k in sorted_keys])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
    
    def plot_by_compression(self, metrics_data: Dict[str, Dict], latency_metric: str, throughput_metric: str, filename: str):
        """Plot.AB for z=n: Compare compression methods"""
        variants = {}
        
        for key in metrics_data.keys():
            if key.startswith('SelfIndex-v1.'):
                compr_type = key.split('.')[1][2]  # Extract z from v1.xyziq
                variants[compr_type] = metrics_data[key]
        
        if not variants:
            print(f"Warning: No variants found for compression comparison")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        compr_labels = {'1': 'None', '2': 'VarByte', '3': 'Zstd'}
        
        sorted_keys = sorted(variants.keys())
        x_pos = np.arange(len(sorted_keys))
        
        # Plot latency
        lat_values = [variants[k][latency_metric] for k in sorted_keys]
        bars1 = ax1.bar(x_pos, lat_values, alpha=0.8, color='steelblue')
        
        for bar, val in zip(bars1, lat_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax1.set_xlabel('Compression Type (z)', fontsize=12)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('Latency by Compression', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([compr_labels[k] for k in sorted_keys])
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot throughput
        thr_values = [variants[k][throughput_metric] for k in sorted_keys]
        bars2 = ax2.bar(x_pos, thr_values, alpha=0.8, color='coral')
        
        for bar, val in zip(bars2, thr_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('Compression Type (z)', fontsize=12)
        ax2.set_ylabel('Queries/Second', fontsize=12)
        ax2.set_title('Throughput by Compression', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([compr_labels[k] for k in sorted_keys])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Plot.AB: Compression Methods Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
    
    def plot_by_skip_pointers(self, metrics_data: Dict[str, Dict], metric_name: str, filename: str):
        """Plot.A for i=0/1: Compare with/without skip pointers (only for TERMatat)"""
        variants = {}
        for key, value in metrics_data.items():
            if key.startswith('SelfIndex-v1.'):
                parts = key.split('.')[1]
                skip = parts[3]  # Extract i from v1.xyziq
                qproc = parts[4]  # Extract q from v1.xyziq
                # Only compare for Term-at-a-time
                if qproc == 'T':
                    variants[skip] = value
        
        if not variants:
            print(f"Warning: No variants found for skip pointers comparison")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        skip_labels = {'0': 'Without Skip Pointers', '1': 'With Skip Pointers'}
        
        sorted_keys = sorted(variants.keys())
        x_pos = np.arange(len(sorted_keys))
        values = [variants[k][metric_name] for k in sorted_keys]
        
        bars = ax.bar(x_pos, values, alpha=0.8, color='green')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Optimization (i)', fontsize=12)
        ax.set_ylabel(f'{metric_name.upper() if metric_name in ["p95", "p99"] else metric_name.capitalize()} (ms)', fontsize=12)
        ax.set_title(f'Plot.A: Impact of Skip Pointers (Term-at-a-Time)', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([skip_labels[k] for k in sorted_keys])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
    
    def plot_by_query_processing(self, metrics_data: Dict[str, Dict], latency_metric: str, memory_metric: str, filename: str):
        """Plot.AC for q=Tn/Dn: Compare Term-at-a-time vs Document-at-a-time"""
        variants = {}
        
        for key in metrics_data.keys():
            if key.startswith('SelfIndex-v1.'):
                parts = key.split('.')[1]
                qproc = parts[4]  # Extract q from v1.xyziq
                skip = parts[3]   # Extract i from v1.xyziq
                
                # Create label with optimization info
                if qproc == 'T':
                    label = f'T{skip}'  # Tn where n is 0 or 1 for skip pointers
                else:
                    label = 'D0'  # Document-at-a-time doesn't use skip pointers
                
                variants[label] = metrics_data[key]
        
        if not variants:
            print(f"Warning: No variants found for query processing comparison")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sort by label to ensure consistent ordering
        sorted_labels = sorted(variants.keys())
        x_pos = np.arange(len(sorted_labels))
        
        # Plot latency
        lat_values = [variants[k][latency_metric] for k in sorted_labels]
        bars1 = ax1.bar(x_pos, lat_values, alpha=0.8, color='steelblue')
        
        for bar, val in zip(bars1, lat_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax1.set_xlabel('Query Processing (q)', fontsize=12)
        ax1.set_ylabel('Latency (ms)', fontsize=12)
        ax1.set_title('Latency by Query Processing', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(sorted_labels)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot memory
        mem_values = [variants[k][memory_metric] for k in sorted_labels]
        bars2 = ax2.bar(x_pos, mem_values, alpha=0.8, color='coral')
        
        for bar, val in zip(bars2, mem_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('Query Processing (q)', fontsize=12)
        ax2.set_ylabel('Memory (MB)', fontsize=12)
        ax2.set_title('Memory by Query Processing', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(sorted_labels)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Plot.AC: Query Processing Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
