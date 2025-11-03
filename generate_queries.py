#!/usr/bin/env python3
"""
Comprehensive Query Set Generator for Plot.C
Generates diverse phrase and boolean queries for fair index comparison.

Query Types:
1. Simple single-term queries
2. Multi-term phrase queries (implicit AND)
3. Boolean AND queries (explicit)
4. Boolean OR queries
5. Boolean NOT queries
6. Complex nested boolean queries
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class QueryTemplate:
    """Template for generating queries with specific properties."""
    name: str
    description: str
    examples: List[str]
    query_type: str  # 'phrase', 'boolean_and', 'boolean_or', 'boolean_not', 'complex'


class PlotCQueryGenerator:
    """
    Generates diverse query set for Plot.C benchmarking.
    
    Ensures fair comparison across Boolean, WordCount, and TF-IDF indices by:
    - Balanced distribution of query types
    - Varied query complexity (1-5 terms)
    - Mix of high/medium/low frequency terms
    - Both phrase and boolean operators
    """
    
    def __init__(self):
        self.templates = []
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize all query templates."""
        
        # 1. Single-term queries (test basic index lookup)
        self.templates.append(QueryTemplate(
            name="Single Term - High Frequency",
            description="Common words with large posting lists",
            query_type="simple",
            examples=[
                "technology", "science", "government", "research", "development",
                "international", "education", "health", "business", "economy",
                "policy", "system", "data", "analysis", "report",
                "environment", "security", "management", "industry", "service"
            ]
        ))
        
        self.templates.append(QueryTemplate(
            name="Single Term - Medium Frequency", 
            description="Moderately common technical terms",
            query_type="simple",
            examples=[
                "algorithm", "quantum", "biodiversity", "cryptocurrency", "genomics",
                "nanotechnology", "neuroscience", "cybersecurity", "biotechnology", "automation",
                "blockchain", "ecosystem", "infrastructure", "sustainability", "innovation",
                "artificial", "digital", "analytics", "optimization", "simulation"
            ]
        ))
        
        self.templates.append(QueryTemplate(
            name="Single Term - Low Frequency",
            description="Rare or specialized terms",
            query_type="simple", 
            examples=[
                "photosynthesis", "anthropology", "metamorphosis", "epistemology", "cryptography",
                "paleontology", "phenomenology", "thermodynamics", "historiography", "semiotics",
                "geopolitics", "macroeconomics", "jurisprudence", "cartography", "cosmology",
                "etymology", "metallurgy", "ornithology", "seismology", "toxicology"
            ]
        ))
        
        # 2. Two-term phrase queries (implicit AND)
        self.templates.append(QueryTemplate(
            name="Two-Term Phrases",
            description="Simple phrase queries with implicit AND",
            query_type="phrase",
            examples=[
                "machine learning", "climate change", "artificial intelligence", "quantum computing",
                "renewable energy", "data science", "public health", "space exploration",
                "social media", "economic growth", "national security", "digital transformation",
                "financial crisis", "electoral system", "supply chain", "mental health",
                "cyber attack", "trade policy", "urban planning", "global warming"
            ]
        ))
        
        # 3. Three-term phrase queries
        self.templates.append(QueryTemplate(
            name="Three-Term Phrases",
            description="More specific phrase queries",
            query_type="phrase",
            examples=[
                "deep learning models", "climate change impact", "public health crisis",
                "economic policy analysis", "renewable energy sources", "artificial neural networks",
                "data privacy protection", "space exploration mission", "global supply chain",
                "quantum physics research", "social justice movement", "financial market stability",
                "cyber security threat", "natural language processing", "urban development planning",
                "international trade agreement", "medical research breakthrough", "environmental sustainability goals",
                "digital currency regulation", "scientific innovation technology"
            ]
        ))
        
        # 4. Boolean AND queries (explicit)
        self.templates.append(QueryTemplate(
            name="Boolean AND Queries",
            description="Explicit AND operator for intersection",
            query_type="boolean_and",
            examples=[
                "artificial AND intelligence", "machine AND learning AND deep",
                "climate AND change AND policy", "renewable AND energy AND solar",
                "data AND science AND analytics", "cyber AND security AND attack",
                "health AND care AND system", "economic AND growth AND development",
                "technology AND innovation AND research", "environmental AND protection AND law",
                "education AND reform AND policy", "financial AND market AND regulation",
                "space AND exploration AND NASA", "quantum AND computing AND algorithm",
                "social AND media AND privacy", "urban AND planning AND infrastructure"
            ]
        ))
        
        # 5. Boolean OR queries (union)
        self.templates.append(QueryTemplate(
            name="Boolean OR Queries",
            description="OR operator for broader recall",
            query_type="boolean_or",
            examples=[
                "machine OR learning", "climate OR environment OR ecology",
                "health OR medical OR healthcare", "economy OR economic OR financial",
                "technology OR innovation OR digital", "security OR safety OR protection",
                "education OR training OR learning", "energy OR power OR renewable",
                "government OR policy OR legislation", "research OR study OR investigation",
                "science OR scientific OR research", "business OR industry OR commerce",
                "data OR analytics OR statistics", "international OR global OR worldwide",
                "development OR progress OR advancement", "system OR infrastructure OR network"
            ]
        ))
        
        # 6. Boolean NOT queries (exclusion)
        self.templates.append(QueryTemplate(
            name="Boolean NOT Queries",
            description="NOT operator for filtering results",
            query_type="boolean_not",
            examples=[
                "technology NOT computer", "health NOT mental",
                "energy NOT nuclear", "science NOT fiction",
                "research NOT medical", "education NOT higher",
                "policy NOT foreign", "economy NOT recession",
                "climate NOT warming", "security NOT cyber",
                "data NOT privacy", "development NOT software",
                "market NOT stock", "government NOT federal",
                "system NOT operating", "business NOT small"
            ]
        ))
        
        # 7. Complex boolean queries (nested operators)
        self.templates.append(QueryTemplate(
            name="Complex Boolean Queries",
            description="Nested boolean expressions",
            query_type="complex",
            examples=[
                "(machine OR deep) AND learning",
                "(climate OR environment) AND (change OR warming)",
                "(artificial AND intelligence) OR (machine AND learning)",
                "technology AND (innovation OR research) NOT military",
                "(health OR medical) AND (research OR study) NOT animal",
                "(renewable OR solar) AND energy NOT nuclear",
                "(data OR information) AND (security OR privacy)",
                "(economic OR financial) AND (growth OR development) NOT recession",
                "(social OR digital) AND media NOT advertising",
                "(quantum OR classical) AND computing NOT simulation",
                "(public OR private) AND (health OR healthcare) NOT insurance",
                "(artificial OR natural) AND intelligence NOT extraterrestrial"
            ]
        ))
        
        # 8. Long-tail queries (very specific)
        self.templates.append(QueryTemplate(
            name="Long-Tail Queries",
            description="Longer, more specific queries",
            query_type="phrase",
            examples=[
                "machine learning deep neural network architecture",
                "climate change environmental impact mitigation strategies",
                "renewable energy solar power grid integration",
                "artificial intelligence natural language processing applications",
                "quantum computing cryptography security implications",
                "blockchain technology distributed ledger financial systems",
                "cybersecurity threat detection prevention response",
                "sustainable development environmental economic social",
                "global health pandemic preparedness response systems",
                "autonomous vehicle safety regulation policy framework"
            ]
        ))
    
    def generate_balanced_query_set(self, total_queries: int = 100) -> Dict:
        """
        Generate a balanced query set with fair distribution.
        
        Args:
            total_queries: Total number of queries to generate
            
        Returns:
            Dict with queries and metadata
        """
        # Calculate queries per template
        queries_per_template = total_queries // len(self.templates)
        remainder = total_queries % len(self.templates)
        
        all_queries = []
        category_breakdown = {}
        
        for i, template in enumerate(self.templates):
            # Allocate extra queries to first templates for remainder
            n_queries = queries_per_template + (1 if i < remainder else 0)
            
            # Sample queries from this template
            if len(template.examples) >= n_queries:
                sampled = random.sample(template.examples, n_queries)
            else:
                # If not enough examples, sample with replacement
                sampled = random.choices(template.examples, k=n_queries)
            
            all_queries.extend(sampled)
            category_breakdown[template.name] = {
                "count": len(sampled),
                "type": template.query_type,
                "description": template.description
            }
        
        # Shuffle to avoid sequential patterns
        random.shuffle(all_queries)
        
        # Calculate type distribution
        type_distribution = {}
        for template in self.templates:
            qtype = template.query_type
            type_distribution[qtype] = type_distribution.get(qtype, 0) + category_breakdown[template.name]["count"]
        
        result = {
            "metadata": {
                "total_queries": len(all_queries),
                "num_templates": len(self.templates),
                "distribution_strategy": "balanced_across_templates",
                "query_types": list(type_distribution.keys()),
                "type_distribution": type_distribution,
                "category_breakdown": category_breakdown
            },
            "queries": all_queries
        }
        
        return result
    
    def generate_justification(self) -> Dict[str, str]:
        """Generate justification for query set design."""
        return {
            "Balanced Distribution": 
                "Queries are evenly distributed across all templates to ensure no bias "
                "toward any particular query type. This gives fair representation to "
                "simple queries, phrase queries, and boolean operators.",
            
            "Query Type Diversity": 
                "Mix of simple terms, phrases (implicit AND), explicit boolean operators "
                "(AND/OR/NOT), and complex nested expressions tests all index capabilities. "
                "Boolean index should handle all types, while TF-IDF excels at ranking phrases.",
            
            "Term Frequency Variation": 
                "Queries include high-frequency (common), medium-frequency (technical), "
                "and low-frequency (rare) terms. This tests index performance across "
                "different posting list sizes - critical for throughput comparison.",
            
            "Query Length Spectrum": 
                "From single-term to 5+ term queries tests scalability of term-at-a-time "
                "processing. Single-term queries benefit from fast path optimization, "
                "while multi-term queries stress accumulator performance.",
            
            "Operator Testing": 
                "Boolean operators (AND/OR/NOT) test set operations vs scoring approaches. "
                "Boolean index uses set union/intersection, while TF-IDF uses score accumulation. "
                "This reveals architectural tradeoffs between the approaches.",
            
            "Long-Tail Coverage": 
                "Specific, multi-concept queries test ranking quality and precision. "
                "These queries should return fewer but more relevant results, "
                "evaluating how well each index handles complex information needs.",
            
            "Fair Comparison Principle": 
                "Every index type gets queries it can handle (simple boolean supports all operators) "
                "while also testing ranking quality where applicable (TF-IDF should rank better "
                "than WordCount). This ensures comparison measures both capability and quality.",
            
            "Real-World Representation": 
                "Query distribution mirrors actual search patterns: mix of simple lookups, "
                "phrase searches, and boolean filters. This makes benchmark results "
                "indicative of real system performance."
        }
    
    def save_query_set(self, output_path: str = "artifacts/query_set.json", num_queries: int = 100):
        """Generate and save query set to file."""
        print(f"\n{'='*80}")
        print("PLOT.C QUERY SET GENERATOR")
        print(f"{'='*80}\n")
        
        # Generate query set
        query_set = self.generate_balanced_query_set(num_queries)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(query_set, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Query set generated: {num_queries} queries")
        print(f"✓ Saved to: {output_file}\n")
        
        # Print summary
        print("Query Distribution:")
        for qtype, count in query_set['metadata']['type_distribution'].items():
            percentage = (count / num_queries) * 100
            print(f"  - {qtype:15} : {count:3} queries ({percentage:5.1f}%)")
        
        print("\nCategory Breakdown:")
        for cat_name, cat_info in query_set['metadata']['category_breakdown'].items():
            print(f"  - {cat_name:30} : {cat_info['count']:3} queries")
        
        print(f"\n{'='*80}")
        print("QUERY SET GENERATION COMPLETE")
        print(f"{'='*80}\n")
        
        return query_set


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Plot.C query set')
    parser.add_argument('--output', default='artifacts/query_set.json',
                       help='Output file path')
    parser.add_argument('--num-queries', type=int, default=100,
                       help='Number of queries to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Generate and save
    generator = PlotCQueryGenerator()
    query_set = generator.save_query_set(args.output, args.num_queries)
    
    # Print sample queries
    print("\nSample Queries:")
    for i, query in enumerate(query_set['queries'][:10], 1):
        print(f"  {i:2}. {query}")
    
    if len(query_set['queries']) > 10:
        print(f"  ... ({len(query_set['queries']) - 10} more queries)")


if __name__ == '__main__':
    main()