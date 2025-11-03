import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path
from dotenv import load_dotenv

from indexing_and_retrieval.data_loader import DataManager
from indexing_and_retrieval.self_index import SelfIndex
from indexing_and_retrieval.metrics import PerformanceMetrics

load_dotenv()

@hydra.main(version_base=None, config_path="config", config_name="config")
def run_experiment(cfg: DictConfig):
    if len(sys.argv) < 2:
        print("Usage: python experiment.py <experiment_name>")
        print("\nAvailable experiments:")
        print("  info_comparison    - Compare BOOLEAN, WORDCOUNT, TFIDF")
        print("  datastore_comparison - Compare CUSTOM, DB1, DB2")
        print("  compression_comparison - Compare NONE, CODE, CLIB")
        print("  query_proc_comparison - Compare TERMatat, DOCatat")
        return
    
    experiment = sys.argv[1]
    
    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    perf_metrics = PerformanceMetrics(metrics_dir)
    test_queries = cfg.metrics.queries
    
    data_manager = DataManager(cfg)
    print("Loading documents...")
    documents = data_manager.load_all_documents()
    
    if experiment == 'info_comparison':
        print("\n=== Information Indexed Comparison ===")
        
        variants = [
            ('BOOLEAN', 'x=1'),
            ('WORDCOUNT', 'x=2'),
            ('TFIDF', 'x=3'),
        ]
        
        indices = {}
        for info, label in variants:
            print(f"\nCreating index: {label} ({info})")
            idx = SelfIndex(cfg, 'SelfIndex', info, 'CUSTOM', 'TERMatat', 'NONE', 'Null')
            idx.create_index(f"exp-{info}", documents)
            indices[label] = idx
        
        all_metrics = {}
        for label, idx in indices.items():
            print(f"\nBenchmarking {label}...")
            latency = perf_metrics.measure_latency(idx.query, test_queries, 5, 50)
            throughput = perf_metrics.measure_throughput(idx.query, test_queries, 5)
            memory = perf_metrics.measure_memory_footprint(idx)
            
            all_metrics[label] = {
                'latency': latency,
                'throughput': throughput,
                'memory': memory
            }
        
        perf_metrics.save_metrics(all_metrics, 'info_comparison.json')
        perf_metrics.plot_latency_comparison(
            {k: v['latency'] for k, v in all_metrics.items()},
            'info_comparison_latency.png'
        )
        perf_metrics.plot_memory_comparison(
            {k: v['memory'] for k, v in all_metrics.items()},
            'info_comparison_memory.png'
        )
    
    elif experiment == 'datastore_comparison':
        print("\n=== Datastore Comparison ===")
        
        variants = [
            ('CUSTOM', 'y=1'),
            ('DB1', 'y=2 (PostgreSQL)'),
            ('DB2', 'y=3 (Redis)'),
        ]
        
        indices = {}
        for dstore, label in variants:
            print(f"\nCreating index: {label} ({dstore})")
            idx = SelfIndex(cfg, 'SelfIndex', 'BOOLEAN', dstore, 'TERMatat', 'NONE', 'Null')
            idx.create_index(f"exp-{dstore}", documents)
            indices[label] = idx
        
        all_metrics = {}
        for label, idx in indices.items():
            print(f"\nBenchmarking {label}...")
            latency = perf_metrics.measure_latency(idx.query, test_queries, 5, 50)
            throughput = perf_metrics.measure_throughput(idx.query, test_queries, 5)
            
            all_metrics[label] = {
                'latency': latency,
                'throughput': throughput
            }
        
        perf_metrics.save_metrics(all_metrics, 'datastore_comparison.json')
        perf_metrics.plot_latency_comparison(
            {k: v['latency'] for k, v in all_metrics.items()},
            'datastore_comparison_latency.png'
        )
        perf_metrics.plot_throughput_comparison(
            {k: v['throughput'] for k, v in all_metrics.items()},
            'datastore_comparison_throughput.png'
        )
    
    elif experiment == 'compression_comparison':
        print("\n=== Compression Comparison ===")
        
        variants = [
            ('NONE', 'z=0'),
            ('CODE', 'z=1 (VarByte)'),
            ('CLIB', 'z=2 (Zstandard)'),
        ]
        
        indices = {}
        for compr, label in variants:
            print(f"\nCreating index: {label} ({compr})")
            idx = SelfIndex(cfg, 'SelfIndex', 'BOOLEAN', 'CUSTOM', 'TERMatat', compr, 'Null')
            idx.create_index(f"exp-{compr}", documents)
            indices[label] = idx
        
        all_metrics = {}
        for label, idx in indices.items():
            print(f"\nBenchmarking {label}...")
            latency = perf_metrics.measure_latency(idx.query, test_queries, 5, 50)
            memory = perf_metrics.measure_memory_footprint(idx)
            
            all_metrics[label] = {
                'latency': latency,
                'memory': memory
            }
        
        perf_metrics.save_metrics(all_metrics, 'compression_comparison.json')
        perf_metrics.plot_latency_comparison(
            {k: v['latency'] for k, v in all_metrics.items()},
            'compression_comparison_latency.png'
        )
        perf_metrics.plot_memory_comparison(
            {k: v['memory'] for k, v in all_metrics.items()},
            'compression_comparison_memory.png'
        )
    
    elif experiment == 'query_proc_comparison':
        print("\n=== Query Processing Comparison ===")
        
        variants = [
            ('TERMatat', 'q=T (Term-at-a-time)'),
            ('DOCatat', 'q=D (Document-at-a-time)'),
        ]
        
        indices = {}
        for qproc, label in variants:
            print(f"\nCreating index: {label} ({qproc})")
            idx = SelfIndex(cfg, 'SelfIndex', 'BOOLEAN', 'CUSTOM', qproc, 'NONE', 'Null')
            idx.create_index(f"exp-{qproc}", documents)
            indices[label] = idx
        
        all_metrics = {}
        for label, idx in indices.items():
            print(f"\nBenchmarking {label}...")
            latency = perf_metrics.measure_latency(idx.query, test_queries, 5, 50)
            throughput = perf_metrics.measure_throughput(idx.query, test_queries, 5)
            
            all_metrics[label] = {
                'latency': latency,
                'throughput': throughput
            }
        
        perf_metrics.save_metrics(all_metrics, 'query_proc_comparison.json')
        perf_metrics.plot_latency_comparison(
            {k: v['latency'] for k, v in all_metrics.items()},
            'query_proc_comparison_latency.png'
        )
        perf_metrics.plot_throughput_comparison(
            {k: v['throughput'] for k, v in all_metrics.items()},
            'query_proc_comparison_throughput.png'
        )
    
    else:
        print(f"Unknown experiment: {experiment}")

if __name__ == "__main__":
    run_experiment()
