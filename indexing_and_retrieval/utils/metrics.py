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
        print(f"  Warmup ({num_warmup} queries)...", end='', flush=True)
        
        # Warmup phase - just a few queries
        for i in range(min(num_warmup, len(queries))):
            query_func(queries[i])
        print(" Done", flush=True)
        
        print(f"  Measuring latency ({num_iterations} iterations)...", end='', flush=True)
        latencies = []
        for i in range(num_iterations):
            query = queries[i % len(queries)]  # Cycle through queries
            start_time = time.perf_counter()
            query_func(query)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
            # Progress indicator
            if (i + 1) % 5 == 0:
                print('.', end='', flush=True)
        
        print(" Done", flush=True)
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
        print(f"Measuring throughput for {duration_seconds} seconds...", end='', flush=True)
        
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
                # Print progress every 10 queries
                if query_count % 10 == 0:
                    print('.', end='', flush=True)
            except Exception as e:
                print(f"\nWarning: Query failed during throughput test: {e}")
                break
        
        print()  # New line after completion
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
    
    def compute_precision_recall_vs_ground_truth(self, index_obj, ground_truth_index, 
                                                  queries: List[str], max_queries: int = 5) -> Dict:
        """
        Compute precision/recall metrics by comparing index results against ground truth (ESIndex).
        Uses only a subset of queries to speed up evaluation.
        
        Args:
            index_obj: The index to evaluate
            ground_truth_index: The ground truth index (typically ESIndex)
            queries: List of queries to test
            max_queries: Maximum number of queries to use (default: 5 for speed)
            
        Returns:
            Dictionary with average precision, recall, and F1 scores
        """
        # Use only first max_queries for speed
        test_queries = queries[:max_queries]
        
        precisions = []
        recalls = []
        f1_scores = []
        
        print(f"    Computing P/R on {len(test_queries)} queries...", end='', flush=True)
        
        for i, query in enumerate(test_queries):
            try:
                # Get results from both indices
                retrieved = set(index_obj.query(query))
                ground_truth = set(ground_truth_index.query(query))
                
                # Compute metrics
                metrics = self.compute_precision_recall(retrieved, ground_truth)
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])
                f1_scores.append(metrics['f1'])
                
                if (i + 1) % 2 == 0:
                    print('.', end='', flush=True)
            except Exception as e:
                print(f"\n    Warning: Query '{query}' failed: {e}")
                continue
        
        print(' Done', flush=True)
        
        return {
            'avg_precision': sum(precisions) / len(precisions) if precisions else 0.0,
            'avg_recall': sum(recalls) / len(recalls) if recalls else 0.0,
            'avg_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            'num_queries': len(precisions)
        }
    
    def save_metrics(self, metrics: Dict, filename: str):
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
    
    def plot_by_info_type(self, latency_metrics: Dict, throughput_metrics: Dict, 
                          memory_metrics: Dict, functional_metrics: Dict, output_file: str):
        """
        Plot comparison by information type (BOOLEAN=1, WORDCOUNT=2, TFIDF=3).
        Each plot shows A: Latency, B: Throughput, C: Memory, D: Precision/Recall.
        Includes ESIndex as baseline comparison.
        """
        # Filter variants by information type
        info_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']
        info_labels = {'BOOLEAN': 'Boolean', 'WORDCOUNT': 'WordCount', 'TFIDF': 'TF-IDF', 'ESIndex': 'ESIndex'}
        
        variants = {}
        # Add SelfIndex variants
        for info in info_types:
            key = f'SelfIndex-v1.{info_types.index(info)+1}110T'  # Base config with varying info type
            if key in latency_metrics:
                variants[info] = {
                    'latency': latency_metrics[key],
                    'throughput': throughput_metrics[key],
                    'memory': memory_metrics[key],
                    'functional': functional_metrics.get(key, {})
                }
        
        # Add ESIndex
        if 'ESIndex' in latency_metrics:
            variants['ESIndex'] = {
                'latency': latency_metrics['ESIndex'],
                'throughput': throughput_metrics['ESIndex'],
                'memory': memory_metrics['ESIndex'],
                'functional': {'avg_precision': 1.0, 'avg_recall': 1.0, 'avg_f1': 1.0}  # Ground truth
            }
        
        if not variants:
            print(f"Warning: No data for info type comparison")
            return
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Plot.C - Comparison by Information Type (vs ESIndex)', fontsize=14, fontweight='bold')
        
        sorted_keys = sorted(variants.keys(), key=lambda x: (x != 'ESIndex', info_types.index(x) if x in info_types else -1))
        x_pos = np.arange(len(sorted_keys))
        
        # Plot A: Latency with P95 and P99
        ax = axes[0, 0]
        means = [variants[k]['latency'].get('mean', 0) for k in sorted_keys]
        p95s = [variants[k]['latency'].get('p95', 0) for k in sorted_keys]
        p99s = [variants[k]['latency'].get('p99', 0) for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, means, width, label='Mean', alpha=0.8, color='steelblue')
        ax.bar(x_pos, p95s, width, label='P95', alpha=0.8, color='orange')
        ax.bar(x_pos + width, p99s, width, label='P99', alpha=0.8, color='red')
        
        ax.set_xlabel('Index Type', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('A: System Response Time (Latency)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([info_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot B: Throughput
        ax = axes[0, 1]
        throughputs = [variants[k]['throughput'].get('queries_per_second', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, throughputs, alpha=0.8, color='green')
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Index Type', fontsize=11)
        ax.set_ylabel('Queries/Second', fontsize=11)
        ax.set_title('B: System Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([info_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot C: Memory Footprint
        ax = axes[1, 0]
        memory = [variants[k]['memory'].get('rss_mb', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, memory, alpha=0.8, color='purple')
        for bar, val in zip(bars, memory):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Index Type', fontsize=11)
        ax.set_ylabel('Memory (MB)', fontsize=11)
        ax.set_title('C: Memory Footprint', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([info_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot D: Precision/Recall/F1 (using ESIndex as ground truth)
        ax = axes[1, 1]
        precisions = [variants[k]['functional'].get('avg_precision', 0) * 100 for k in sorted_keys]
        recalls = [variants[k]['functional'].get('avg_recall', 0) * 100 for k in sorted_keys]
        f1s = [variants[k]['functional'].get('avg_f1', 0) * 100 for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8, color='cyan')
        ax.bar(x_pos, recalls, width, label='Recall', alpha=0.8, color='magenta')
        ax.bar(x_pos + width, f1s, width, label='F1-Score', alpha=0.8, color='gold')
        
        ax.set_xlabel('Index Type', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('D: Functional Metrics (vs ESIndex)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([info_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.set_ylim(95, 100)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {output_file}")
    
    def plot_by_datastore(self, all_latency: Dict, all_throughput: Dict, all_memory: Dict, 
                          all_functional: Dict, filename: str):
        """Plot.A for y=n: Compare datastore choices (CUSTOM, DB1, DB2)
        Shows all 4 metrics: A (latency), B (throughput), C (memory), D (functional metrics)"""
        
        variants = {}
        for key in all_latency.keys():
            if key.startswith('SelfIndex-v1.'):
                dstore_type = key.split('.')[1][1]  # Extract y from v1.xyziq
                if dstore_type not in variants:
                    variants[dstore_type] = {
                        'latency': all_latency.get(key, {}),
                        'throughput': all_throughput.get(key, {}),
                        'memory': all_memory.get(key, {}),
                        'functional': all_functional.get(key, {})
                    }
        
        # Add ESIndex
        if 'ESIndex' in all_latency:
            variants['ESIndex'] = {
                'latency': all_latency['ESIndex'],
                'throughput': all_throughput['ESIndex'],
                'memory': all_memory['ESIndex'],
                'functional': {'avg_precision': 1.0, 'avg_recall': 1.0, 'avg_f1': 1.0}
            }
        
        if not variants:
            print(f"Warning: No variants found for datastore comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot.A - Comparison by Datastore Type (vs ESIndex)', fontsize=14, fontweight='bold')
        
        dstore_labels = {'1': 'Custom Disk', '2': 'PostgreSQL', '3': 'Redis', 'ESIndex': 'ESIndex'}
        sorted_keys = sorted(variants.keys(), key=lambda x: (x != 'ESIndex', x))
        x_pos = np.arange(len(sorted_keys))
        
        # Plot A: Latency
        ax = axes[0, 0]
        means = [variants[k]['latency'].get('mean', 0) for k in sorted_keys]
        p95s = [variants[k]['latency'].get('p95', 0) for k in sorted_keys]
        p99s = [variants[k]['latency'].get('p99', 0) for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, means, width, label='Mean', alpha=0.8, color='steelblue')
        ax.bar(x_pos, p95s, width, label='P95', alpha=0.8, color='orange')
        ax.bar(x_pos + width, p99s, width, label='P99', alpha=0.8, color='red')
        
        ax.set_xlabel('Datastore Type', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('A: System Response Time (Latency)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([dstore_labels[k] for k in sorted_keys])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot B: Throughput
        ax = axes[0, 1]
        throughputs = [variants[k]['throughput'].get('queries_per_second', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, throughputs, alpha=0.8, color='green')
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Datastore Type', fontsize=11)
        ax.set_ylabel('Queries/Second', fontsize=11)
        ax.set_title('B: System Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([dstore_labels[k] for k in sorted_keys])
        ax.grid(axis='y', alpha=0.3)
        
        # Plot C: Memory Footprint
        ax = axes[1, 0]
        memory = [variants[k]['memory'].get('rss_mb', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, memory, alpha=0.8, color='purple')
        for bar, val in zip(bars, memory):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Datastore Type', fontsize=11)
        ax.set_ylabel('Memory (MB)', fontsize=11)
        ax.set_title('C: Memory Footprint', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([dstore_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot D: Functional Metrics
        ax = axes[1, 1]
        precisions = [variants[k]['functional'].get('avg_precision', 0) * 100 for k in sorted_keys]
        recalls = [variants[k]['functional'].get('avg_recall', 0) * 100 for k in sorted_keys]
        f1s = [variants[k]['functional'].get('avg_f1', 0) * 100 for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8, color='cyan')
        ax.bar(x_pos, recalls, width, label='Recall', alpha=0.8, color='magenta')
        ax.bar(x_pos + width, f1s, width, label='F1-Score', alpha=0.8, color='gold')
        
        ax.set_xlabel('Datastore Type', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('D: Functional Metrics (vs ESIndex)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([dstore_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.set_ylim(95, 100)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
    
    def plot_by_compression(self, all_latency: Dict, all_throughput: Dict, all_memory: Dict, 
                            all_functional: Dict, filename: str):
        """Plot.AB for z=n: Compare compression methods (NONE, CODE, CLIB)
        Shows all 4 metrics: A (latency), B (throughput), C (memory), D (functional metrics)"""
        
        variants = {}
        for key in all_latency.keys():
            if key.startswith('SelfIndex-v1.'):
                compr_type = key.split('.')[1][2]  # Extract z from v1.xyziq
                if compr_type not in variants:
                    variants[compr_type] = {
                        'latency': all_latency.get(key, {}),
                        'throughput': all_throughput.get(key, {}),
                        'memory': all_memory.get(key, {}),
                        'functional': all_functional.get(key, {})
                    }
        
        # Add ESIndex
        if 'ESIndex' in all_latency:
            variants['ESIndex'] = {
                'latency': all_latency['ESIndex'],
                'throughput': all_throughput['ESIndex'],
                'memory': all_memory['ESIndex'],
                'functional': {'avg_precision': 1.0, 'avg_recall': 1.0, 'avg_f1': 1.0}
            }
        
        if not variants:
            print(f"Warning: No variants found for compression comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot.AB - Comparison by Compression Method (vs ESIndex)', fontsize=14, fontweight='bold')
        
        compr_labels = {'1': 'None', '2': 'VarByte', '3': 'Zstd', 'ESIndex': 'ESIndex'}
        sorted_keys = sorted(variants.keys(), key=lambda x: (x != 'ESIndex', x))
        x_pos = np.arange(len(sorted_keys))
        
        # Plot A: Latency
        ax = axes[0, 0]
        means = [variants[k]['latency'].get('mean', 0) for k in sorted_keys]
        p95s = [variants[k]['latency'].get('p95', 0) for k in sorted_keys]
        p99s = [variants[k]['latency'].get('p99', 0) for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, means, width, label='Mean', alpha=0.8, color='steelblue')
        ax.bar(x_pos, p95s, width, label='P95', alpha=0.8, color='orange')
        ax.bar(x_pos + width, p99s, width, label='P99', alpha=0.8, color='red')
        
        ax.set_xlabel('Compression Type', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('A: System Response Time (Latency)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([compr_labels[k] for k in sorted_keys])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot B: Throughput
        ax = axes[0, 1]
        throughputs = [variants[k]['throughput'].get('queries_per_second', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, throughputs, alpha=0.8, color='green')
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Compression Type', fontsize=11)
        ax.set_ylabel('Queries/Second', fontsize=11)
        ax.set_title('B: System Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([compr_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot C: Memory Footprint
        ax = axes[1, 0]
        memory = [variants[k]['memory'].get('rss_mb', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, memory, alpha=0.8, color='purple')
        for bar, val in zip(bars, memory):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Compression Type', fontsize=11)
        ax.set_ylabel('Memory (MB)', fontsize=11)
        ax.set_title('C: Memory Footprint', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([compr_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot D: Functional Metrics
        ax = axes[1, 1]
        precisions = [variants[k]['functional'].get('avg_precision', 0) * 100 for k in sorted_keys]
        recalls = [variants[k]['functional'].get('avg_recall', 0) * 100 for k in sorted_keys]
        f1s = [variants[k]['functional'].get('avg_f1', 0) * 100 for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8, color='cyan')
        ax.bar(x_pos, recalls, width, label='Recall', alpha=0.8, color='magenta')
        ax.bar(x_pos + width, f1s, width, label='F1-Score', alpha=0.8, color='gold')
        
        ax.set_xlabel('Compression Type', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('D: Functional Metrics (vs ESIndex)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([compr_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.set_ylim(95, 100)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
    
    def plot_by_skip_pointers(self, all_latency: Dict, all_throughput: Dict, all_memory: Dict, 
                              all_functional: Dict, filename: str):
        """Plot.A for i=0/1: Compare with/without skip pointers (only for TERMatat)
        Shows all 4 metrics: A (latency), B (throughput), C (memory), D (functional metrics)"""
        
        variants = {}
        for key in all_latency.keys():
            if key.startswith('SelfIndex-v1.'):
                parts = key.split('.')[1]
                skip = parts[3]  # Extract i from v1.xyziq
                qproc = parts[4]  # Extract q from v1.xyziq
                # Only compare for Term-at-a-time
                if qproc == 'T':
                    if skip not in variants:
                        variants[skip] = {
                            'latency': all_latency.get(key, {}),
                            'throughput': all_throughput.get(key, {}),
                            'memory': all_memory.get(key, {}),
                            'functional': all_functional.get(key, {})
                        }
        
        # Add ESIndex
        if 'ESIndex' in all_latency:
            variants['ESIndex'] = {
                'latency': all_latency['ESIndex'],
                'throughput': all_throughput['ESIndex'],
                'memory': all_memory['ESIndex'],
                'functional': {'avg_precision': 1.0, 'avg_recall': 1.0, 'avg_f1': 1.0}
            }
        
        if not variants:
            print(f"Warning: No variants found for skip pointers comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot.A - Comparison with/without Skip Pointers (vs ESIndex)', fontsize=14, fontweight='bold')
        
        skip_labels = {'0': 'No Skip Pointers', '1': 'Skip Pointers', 'ESIndex': 'ESIndex'}
        sorted_keys = sorted(variants.keys(), key=lambda x: (x != 'ESIndex', x))
        x_pos = np.arange(len(sorted_keys))
        
        # Plot A: Latency
        ax = axes[0, 0]
        means = [variants[k]['latency'].get('mean', 0) for k in sorted_keys]
        p95s = [variants[k]['latency'].get('p95', 0) for k in sorted_keys]
        p99s = [variants[k]['latency'].get('p99', 0) for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, means, width, label='Mean', alpha=0.8, color='steelblue')
        ax.bar(x_pos, p95s, width, label='P95', alpha=0.8, color='orange')
        ax.bar(x_pos + width, p99s, width, label='P99', alpha=0.8, color='red')
        
        ax.set_xlabel('Skip Pointers', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('A: System Response Time (Latency)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([skip_labels.get(k, k) for k in sorted_keys])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot B: Throughput
        ax = axes[0, 1]
        throughputs = [variants[k]['throughput'].get('queries_per_second', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, throughputs, alpha=0.8, color='green')
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Skip Pointers', fontsize=11)
        ax.set_ylabel('Queries/Second', fontsize=11)
        ax.set_title('B: System Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([skip_labels.get(k, k) for k in sorted_keys])
        ax.grid(axis='y', alpha=0.3)
        
        # Plot C: Memory Footprint
        ax = axes[1, 0]
        memory = [variants[k]['memory'].get('rss_mb', 0) for k in sorted_keys]
        bars = ax.bar(x_pos, memory, alpha=0.8, color='purple')
        for bar, val in zip(bars, memory):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Skip Pointers', fontsize=11)
        ax.set_ylabel('Memory (MB)', fontsize=11)
        ax.set_title('C: Memory Footprint', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([skip_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot D: Functional Metrics
        ax = axes[1, 1]
        precisions = [variants[k]['functional'].get('avg_precision', 0) * 100 for k in sorted_keys]
        recalls = [variants[k]['functional'].get('avg_recall', 0) * 100 for k in sorted_keys]
        f1s = [variants[k]['functional'].get('avg_f1', 0) * 100 for k in sorted_keys]
        
        width = 0.25
        ax.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8, color='cyan')
        ax.bar(x_pos, recalls, width, label='Recall', alpha=0.8, color='magenta')
        ax.bar(x_pos + width, f1s, width, label='F1-Score', alpha=0.8, color='gold')
        
        ax.set_xlabel('Skip Pointers', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('D: Functional Metrics (vs ESIndex)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([skip_labels.get(k, k) for k in sorted_keys], rotation=15, ha='right')
        ax.set_ylim(95, 100)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
    
    def plot_by_query_processing(self, all_latency: Dict, all_throughput: Dict, all_memory: Dict, 
                                  all_functional: Dict, filename: str):
        """Plot.AC for q=T/D: Compare Term-at-a-time vs Document-at-a-time
        Shows all 4 metrics: A (latency), B (throughput), C (memory), D (functional metrics)"""
        
        variants = {}
        for key in all_latency.keys():
            if key.startswith('SelfIndex-v1.'):
                parts = key.split('.')[1]
                qproc = parts[4]  # Extract q from v1.xyziq
                
                label = 'Term-at-a-time' if qproc == 'T' else 'Document-at-a-time'
                
                if label not in variants:
                    variants[label] = {
                        'latency': all_latency.get(key, {}),
                        'throughput': all_throughput.get(key, {}),
                        'memory': all_memory.get(key, {}),
                        'functional': all_functional.get(key, {})
                    }
        
        # Add ESIndex
        if 'ESIndex' in all_latency:
            variants['ESIndex'] = {
                'latency': all_latency['ESIndex'],
                'throughput': all_throughput['ESIndex'],
                'memory': all_memory['ESIndex'],
                'functional': {'avg_precision': 1.0, 'avg_recall': 1.0, 'avg_f1': 1.0}
            }
        
        if not variants:
            print(f"Warning: No variants found for query processing comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Plot.AC - Comparison by Query Processing (vs ESIndex)', fontsize=14, fontweight='bold')
        
        sorted_labels = sorted(variants.keys(), key=lambda x: (x != 'ESIndex', x))
        x_pos = np.arange(len(sorted_labels))
        
        # Plot A: Latency
        ax = axes[0, 0]
        means = [variants[k]['latency'].get('mean', 0) for k in sorted_labels]
        p95s = [variants[k]['latency'].get('p95', 0) for k in sorted_labels]
        p99s = [variants[k]['latency'].get('p99', 0) for k in sorted_labels]
        
        width = 0.25
        ax.bar(x_pos - width, means, width, label='Mean', alpha=0.8, color='steelblue')
        ax.bar(x_pos, p95s, width, label='P95', alpha=0.8, color='orange')
        ax.bar(x_pos + width, p99s, width, label='P99', alpha=0.8, color='red')
        
        ax.set_xlabel('Query Processing Strategy', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('A: System Response Time (Latency)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_labels, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot B: Throughput
        ax = axes[0, 1]
        throughputs = [variants[k]['throughput'].get('queries_per_second', 0) for k in sorted_labels]
        bars = ax.bar(x_pos, throughputs, alpha=0.8, color='green')
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Query Processing Strategy', fontsize=11)
        ax.set_ylabel('Queries/Second', fontsize=11)
        ax.set_title('B: System Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot C: Memory Footprint
        ax = axes[1, 0]
        memory = [variants[k]['memory'].get('rss_mb', 0) for k in sorted_labels]
        bars = ax.bar(x_pos, memory, alpha=0.8, color='purple')
        for bar, val in zip(bars, memory):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Query Processing Strategy', fontsize=11)
        ax.set_ylabel('Memory (MB)', fontsize=11)
        ax.set_title('C: Memory Footprint', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot D: Functional Metrics
        ax = axes[1, 1]
        precisions = [variants[k]['functional'].get('avg_precision', 0) * 100 for k in sorted_labels]
        recalls = [variants[k]['functional'].get('avg_recall', 0) * 100 for k in sorted_labels]
        f1s = [variants[k]['functional'].get('avg_f1', 0) * 100 for k in sorted_labels]
        
        width = 0.25
        ax.bar(x_pos - width, precisions, width, label='Precision', alpha=0.8, color='cyan')
        ax.bar(x_pos, recalls, width, label='Recall', alpha=0.8, color='magenta')
        ax.bar(x_pos + width, f1s, width, label='F1-Score', alpha=0.8, color='gold')
        
        ax.set_xlabel('Query Processing Strategy', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title('D: Functional Metrics (vs ESIndex)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_labels, rotation=15, ha='right')
        ax.set_ylim(95, 100)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {filename}")
