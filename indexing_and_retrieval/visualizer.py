import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import numpy as np

class FrequencyAnalyzer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_frequency_distribution(self, 
                                   freq_counter: Counter,
                                   title: str,
                                   filename: str,
                                   top_n: int = 50,
                                   log_scale: bool = True):
        most_common = freq_counter.most_common(top_n)
        words, counts = zip(*most_common) if most_common else ([], [])
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(words)), counts)
        
        for i, bar in enumerate(bars):
            if i < 10:
                bar.set_color('#e74c3c')
            elif i < 25:
                bar.set_color('#3498db')
            else:
                bar.set_color('#95a5a6')
        
        plt.xlabel('Terms', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        
        if log_scale:
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filename}")
    
    def plot_comparison(self,
                       freq_raw: Counter,
                       freq_processed: Counter,
                       filename: str,
                       top_n: int = 30):
        raw_common = dict(freq_raw.most_common(top_n))
        proc_common = dict(freq_processed.most_common(top_n))
        
        all_terms = sorted(set(raw_common.keys()) | set(proc_common.keys()))
        raw_counts = [raw_common.get(t, 0) for t in all_terms]
        proc_counts = [proc_common.get(t, 0) for t in all_terms]
        
        x = np.arange(len(all_terms))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(16, 8))
        bars1 = ax.bar(x - width/2, raw_counts, width, label='Raw', alpha=0.8, color='#3498db')
        bars2 = ax.bar(x + width/2, proc_counts, width, label='Preprocessed', alpha=0.8, color='#e74c3c')
        
        ax.set_xlabel('Terms', fontsize=12)
        ax.set_ylabel('Frequency (log scale)', fontsize=12)
        ax.set_title('Term Frequency: Raw vs Preprocessed', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_terms, rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot: {filename}")
    
    def plot_zipf_distribution(self,
                              freq_counter: Counter,
                              filename: str):
        sorted_counts = sorted(freq_counter.values(), reverse=True)
        ranks = range(1, len(sorted_counts) + 1)
        
        plt.figure(figsize=(12, 8))
        plt.loglog(ranks, sorted_counts, 'b-', alpha=0.6, linewidth=2)
        plt.xlabel('Rank (log scale)', fontsize=12)
        plt.ylabel('Frequency (log scale)', fontsize=12)
        plt.title("Zipf's Law Distribution", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Zipf plot: {filename}")
