import hydra
from omegaconf import DictConfig
from pathlib import Path
from dotenv import load_dotenv

from indexing_and_retrieval.data_loader import DataManager
from indexing_and_retrieval.preprocessing import TextPreprocessor
from indexing_and_retrieval.visualizer import FrequencyAnalyzer
from indexing_and_retrieval.es_index import ESIndex
from indexing_and_retrieval.self_index import SelfIndex
from indexing_and_retrieval.metrics import PerformanceMetrics

load_dotenv()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("=" * 80)
    print("IRE Assignment: Indexing and Retrieval")
    print("=" * 80)
    
    plots_dir = Path(cfg.paths.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    data_manager = DataManager(cfg)
    preprocessor = TextPreprocessor(cfg.preprocessing)
    freq_analyzer = FrequencyAnalyzer(plots_dir)
    perf_metrics = PerformanceMetrics(metrics_dir)
    
    print("\n[1/6] Loading documents...")
    documents = data_manager.load_all_documents()
    
    print(f"\n[2/6] Analyzing word frequencies...")
    doc_texts = [text for _, text in documents]
    
    print("Computing raw frequencies...")
    freq_raw = preprocessor.compute_term_frequencies(doc_texts, preprocess=False)
    
    print("Computing preprocessed frequencies...")
    freq_processed = preprocessor.compute_term_frequencies(doc_texts, preprocess=True)
    
    print("Generating frequency plots...")
    freq_analyzer.plot_frequency_distribution(
        freq_raw, 
        "Word Frequency Distribution (Raw)", 
        "frequency_raw.png",
        top_n=cfg.preprocessing.plots.top_n,
        log_scale=cfg.preprocessing.plots.log_scale
    )
    
    freq_analyzer.plot_frequency_distribution(
        freq_processed,
        "Word Frequency Distribution (Preprocessed)",
        "frequency_preprocessed.png",
        top_n=cfg.preprocessing.plots.top_n,
        log_scale=cfg.preprocessing.plots.log_scale
    )
    
    freq_analyzer.plot_comparison(
        freq_raw,
        freq_processed,
        "frequency_comparison.png",
        top_n=30
    )
    
    freq_analyzer.plot_zipf_distribution(
        freq_processed,
        "zipf_distribution.png"
    )
    
    print(f"\n[3/6] Creating Elasticsearch index...")
    es_index = ESIndex(cfg)
    es_index.create_index("esindex-v1.0", documents)
    
    print(f"\n[4/6] Benchmarking Elasticsearch...")
    test_queries = cfg.metrics.queries
    
    all_latency_metrics = {}
    all_throughput_metrics = {}
    all_memory_metrics = {}
    all_functional_metrics = {}
    all_metrics = {}
    
    print("\n--- Benchmarking Elasticsearch ---")
    latency_es = perf_metrics.measure_latency(
        es_index.query, 
        test_queries,
        num_warmup=cfg.metrics.num_warmup,
        num_iterations=cfg.metrics.num_iterations
    )
    all_latency_metrics['ESIndex'] = latency_es
    print(f"Latency - Mean: {latency_es['mean']:.2f}ms, P95: {latency_es['p95']:.2f}ms, P99: {latency_es['p99']:.2f}ms")
    
    throughput_es = perf_metrics.measure_throughput(
        es_index.query,
        test_queries,
        duration_seconds=cfg.metrics.throughput_duration
    )
    all_throughput_metrics['ESIndex'] = throughput_es
    print(f"Throughput: {throughput_es['queries_per_second']:.2f} queries/sec")
    
    memory_es = perf_metrics.measure_memory_footprint(es_index)
    all_memory_metrics['ESIndex'] = memory_es
    print(f"Memory: {memory_es['rss_mb']:.2f} MB")
    
    print(f"\n[5/6] Creating and benchmarking targeted SelfIndex variants...")
    
    # Define base configuration
    BASE_CONFIG = {
        'info': 'BOOLEAN',
        'dstore': 'CUSTOM',
        'compr': 'NONE',
        'qproc': 'TERMatat',
        'optim': 'Null'
    }
    
    def create_and_benchmark_variant(info, dstore, compr, qproc, optim, label):
        """Helper function to create and benchmark a single variant"""
        print(f"\n--- {label} ---", flush=True)
        print(f"Creating SelfIndex: info={info}, dstore={dstore}, compr={compr}, qproc={qproc}, optim={optim}", flush=True)
        
        idx = SelfIndex(cfg, 'SelfIndex', info, dstore, qproc, compr, optim)
        idx.create_index(f"selfindex-{info}-{dstore}-{compr}-{qproc}-{optim}", documents)
        
        idx_key = idx.identifier_short
        print(f"Benchmarking {idx_key}...", flush=True)
        
        # Use fewer iterations for SelfIndex to speed up benchmarking on large datasets
        num_iters = max(10, cfg.metrics.num_iterations // 2)
        print(f"  → Measuring latency ({num_iters} iterations)...", flush=True)
        latency = perf_metrics.measure_latency(
            idx.query,
            test_queries,
            num_warmup=cfg.metrics.num_warmup,
            num_iterations=num_iters
        )
        all_latency_metrics[idx_key] = latency
        print(f"  ✓ Latency - Mean: {latency['mean']:.2f}ms, P95: {latency['p95']:.2f}ms, P99: {latency['p99']:.2f}ms", flush=True)
        
        print(f"  → Measuring throughput ({cfg.metrics.throughput_duration}s)...", flush=True)
        throughput = perf_metrics.measure_throughput(
            idx.query,
            test_queries,
            duration_seconds=cfg.metrics.throughput_duration
        )
        all_throughput_metrics[idx_key] = throughput
        print(f"  ✓ Throughput: {throughput['queries_per_second']:.2f} queries/sec", flush=True)
        
        print(f"  → Measuring memory footprint...", flush=True)
        memory = perf_metrics.measure_memory_footprint(idx)
        all_memory_metrics[idx_key] = memory
        print(f"  ✓ Memory: {memory['rss_mb']:.2f} MB", flush=True)
        
        print(f"  → Computing precision/recall vs ESIndex...", flush=True)
        functional = perf_metrics.compute_precision_recall_vs_ground_truth(idx, es_index, test_queries)
        all_functional_metrics[idx_key] = functional
        print(f"  ✓ Precision: {functional['avg_precision']*100:.1f}%, Recall: {functional['avg_recall']*100:.1f}%, F1: {functional['avg_f1']*100:.1f}%", flush=True)
        
        # Close datastore to free memory
        if hasattr(idx, 'datastore') and idx.datastore:
            idx.datastore.close()
        
        # Clear the index from memory
        del idx
        
        import gc
        gc.collect()
        
        return idx_key
    
    # Plot.C: Vary information type (x=1,2,3)
    print("\n=== Plot.C: Information Type Comparison ===")
    for info in ['BOOLEAN', 'WORDCOUNT', 'TFIDF']:
        create_and_benchmark_variant(
            info, BASE_CONFIG['dstore'], BASE_CONFIG['compr'], 
            BASE_CONFIG['qproc'], BASE_CONFIG['optim'],
            f"Plot.C - Info Type: {info}"
        )
    
    # Plot.A (datastores): Vary datastore type (y=1,2,3)
    print("\n=== Plot.A: Datastore Comparison ===")
    for dstore in ['CUSTOM', 'DB1', 'DB2']:
        create_and_benchmark_variant(
            BASE_CONFIG['info'], dstore, BASE_CONFIG['compr'],
            BASE_CONFIG['qproc'], BASE_CONFIG['optim'],
            f"Plot.A - Datastore: {dstore}"
        )
    
    # Plot.AB: Vary compression (z=1,2,3)
    print("\n=== Plot.AB: Compression Comparison ===")
    for compr in ['NONE', 'CODE', 'CLIB']:
        create_and_benchmark_variant(
            BASE_CONFIG['info'], BASE_CONFIG['dstore'], compr,
            BASE_CONFIG['qproc'], BASE_CONFIG['optim'],
            f"Plot.AB - Compression: {compr}"
        )
    
    # Plot.A (skip pointers): Vary optimization (i=0/1) for Term-at-a-time
    print("\n=== Plot.A: Skip Pointers Impact ===")
    for optim in ['Null', 'Skipping']:
        create_and_benchmark_variant(
            BASE_CONFIG['info'], BASE_CONFIG['dstore'], BASE_CONFIG['compr'],
            'TERMatat', optim,
            f"Plot.A - Skip Pointers: {optim}"
        )
    
    # Plot.AC: Vary query processing (q=T/D) with and without optimizations
    print("\n=== Plot.AC: Query Processing Comparison ===")
    # Term-at-a-time without skip pointers
    create_and_benchmark_variant(
        BASE_CONFIG['info'], BASE_CONFIG['dstore'], BASE_CONFIG['compr'],
        'TERMatat', 'Null',
        "Plot.AC - Term-at-a-time (no skipping)"
    )
    # Term-at-a-time with skip pointers
    create_and_benchmark_variant(
        BASE_CONFIG['info'], BASE_CONFIG['dstore'], BASE_CONFIG['compr'],
        'TERMatat', 'Skipping',
        "Plot.AC - Term-at-a-time (with skipping)"
    )
    # Document-at-a-time (skip pointers not applicable)
    create_and_benchmark_variant(
        BASE_CONFIG['info'], BASE_CONFIG['dstore'], BASE_CONFIG['compr'],
        'DOCatat', 'Null',
        "Plot.AC - Document-at-a-time"
    )
    
    print(f"\n[6/6] Generating assignment-specific plots and saving metrics...")
    
    # Combine all metrics into a single dictionary with the index identifier as key
    all_metrics = {}
    for idx_key in all_latency_metrics.keys():
        all_metrics[idx_key] = {
            **all_latency_metrics[idx_key],
            'queries_per_second': all_throughput_metrics[idx_key]['queries_per_second'],
            'rss_mb': all_memory_metrics[idx_key]['rss_mb']
        }
    
    # Generate assignment-specific plots (each plot shows A, B, C, D metrics)
    # Plot.C: Comparison by information type (x=1,2,3)
    perf_metrics.plot_by_info_type(all_latency_metrics, all_throughput_metrics, all_memory_metrics, 
                                   all_functional_metrics, 'plot_c_comparison_by_info_type.png')
    
    # Plot.A: Comparison by datastore type (y=1,2,3)
    perf_metrics.plot_by_datastore(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                   all_functional_metrics, 'plot_a_comparison_by_datastore.png')
    
    # Plot.AB: Comparison by compression (z=1,2,3)
    perf_metrics.plot_by_compression(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                     all_functional_metrics, 'plot_ab_comparison_by_compression.png')
    
    # Plot.A: Comparison with/without skip pointers (i=0/1)
    perf_metrics.plot_by_skip_pointers(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                       all_functional_metrics, 'plot_a_comparison_skip_pointers.png')
    
    # Plot.AC: Comparison by query processing (q=T/D)
    perf_metrics.plot_by_query_processing(all_latency_metrics, all_throughput_metrics, all_memory_metrics,
                                          all_functional_metrics, 'plot_ac_comparison_query_processing.png')
    
    # Save all metrics to JSON
    perf_metrics.save_metrics({
        'latency': all_latency_metrics,
        'throughput': all_throughput_metrics,
        'memory': all_memory_metrics,
        'functional': all_functional_metrics
    }, "all_metrics.json")
    
    print("\n" + "=" * 80)
    print("Assignment completed successfully!")
    print(f"Plots saved to: {plots_dir.absolute()}")
    print(f"Metrics saved to: {metrics_dir.absolute()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
