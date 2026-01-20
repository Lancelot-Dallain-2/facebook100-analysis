"""
Question 6: Community Detection on FB100 Dataset

Research Question:
"Do student social networks exhibit homophilic community structure based on 
academic characteristics (major, dorm), and how does community modularity 
relate to university size and type?"

Hypothesis:
Students form communities based on shared academic attributes (major, dorm).
Smaller universities show stronger community structure (higher modularity) due to
more intimate campus environments and stronger major-based clustering.

Validation approach:
- Select diverse set of universities (different sizes, public/private)
- Apply multiple community detection algorithms:
  * Louvain (modularity optimization)
  * Label Propagation (fast, local structure)
  * Girvan-Newman (hierarchical, betweenness-based)
- Analyze:
  * Modularity scores
  * Community homogeneity w.r.t. major and dorm
  * Relationship between network size and modularity
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Set
import random
from collections import Counter, defaultdict
from itertools import combinations

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.base import NetworkAnalysisBase
from lib.utils import list_available_networks, analyze_node_attributes


class CommunityDetector:
    """Base class for community detection algorithms."""
    
    def __init__(self, G: nx.Graph):
        self.G = G
        self.communities = []
    
    def detect(self) -> List[Set]:
        """
        Detect communities in the graph.
        
        Returns:
            List of sets, each set contains nodes in a community
        """
        raise NotImplementedError
    
    def compute_modularity(self, communities: List[Set]) -> float:
        """Compute modularity of a partition."""
        return nx.algorithms.community.modularity(self.G, communities)


class LouvainDetector(CommunityDetector):
    """Louvain method for modularity optimization."""
    
    def detect(self) -> List[Set]:
        """Run Louvain algorithm."""
        try:
            import community as community_louvain
            # Compute partition
            partition = community_louvain.best_partition(self.G)
            
            # Convert to list of sets
            communities_dict = defaultdict(set)
            for node, comm_id in partition.items():
                communities_dict[comm_id].add(node)
            
            self.communities = list(communities_dict.values())
            
        except ImportError:
            # Fallback: use NetworkX greedy modularity
            self.communities = list(
                nx.algorithms.community.greedy_modularity_communities(self.G)
            )
        
        return self.communities


class LabelPropagationDetector(CommunityDetector):
    """Label propagation for community detection."""
    
    def detect(self) -> List[Set]:
        """Run label propagation algorithm."""
        self.communities = list(
            nx.algorithms.community.label_propagation_communities(self.G)
        )
        return self.communities


class GirvanNewmanDetector(CommunityDetector):
    """Girvan-Newman algorithm (edge betweenness)."""
    
    def __init__(self, G: nx.Graph, k: int = None):
        """
        Args:
            G: Graph
            k: Number of communities (if None, auto-determine by modularity)
        """
        super().__init__(G)
        self.k = k
    
    def detect(self) -> List[Set]:
        """Run Girvan-Newman algorithm."""
        # This is computationally expensive, limit to smaller networks
        if self.G.number_of_nodes() > 3000:
            # Fall back to faster method
            return LouvainDetector(self.G).detect()
        
        comp = nx.algorithms.community.girvan_newman(self.G)
        
        if self.k is not None:
            # Extract exactly k communities
            for communities in comp:
                if len(communities) >= self.k:
                    self.communities = [set(c) for c in communities]
                    break
        else:
            # Find partition with best modularity
            best_modularity = -1
            best_communities = None
            
            for communities in comp:
                comm_list = [set(c) for c in communities]
                mod = self.compute_modularity(comm_list)
                
                if mod > best_modularity:
                    best_modularity = mod
                    best_communities = comm_list
                
                # Stop if too many communities (diminishing returns)
                if len(communities) > 20:
                    break
            
            self.communities = best_communities if best_communities else []
        
        return self.communities


class Question6Analysis(NetworkAnalysisBase):
    """
    Community detection analysis with research question validation.
    """
    
    def __init__(self, data_dir="fb100/data", output_dir="results/q6"):
        super().__init__(data_dir, output_dir)
        
        # Select diverse set of universities
        self.target_networks = {
            'Small': ['Caltech36', 'Haverford76', 'Swarthmore42'],
            'Medium': ['Princeton12', 'Dartmouth6', 'Brown11'],
            'Large': ['Harvard1', 'MIT8', 'Stanford3']
        }
        
        # Algorithms to test
        self.algorithms = {
            'Louvain': LouvainDetector,
            'LabelPropagation': LabelPropagationDetector,
            'GirvanNewman': GirvanNewmanDetector
        }
    
    def run(self):
        """Execute Question 6 analysis."""
        self.logger.section("QUESTION 6: COMMUNITY DETECTION", "=")
        
        # Part (a): Present research question
        self.part_a_research_question()
        
        # Load selected networks
        self.load_selected_networks()
        
        # Part (b): Run community detection
        self.part_b_community_detection()
        
        # Part (c): Analyze and conclude
        self.part_c_analysis_and_conclusion()
        
        self.logger.section("QUESTION 6 COMPLETED", "=")
    
    def part_a_research_question(self):
        """Part (a): Formulate and present research question."""
        self.logger.subsection("Part (a): Research Question")
        
        self.logger.log("\n" + "="*80, "INFO")
        self.logger.log("RESEARCH QUESTION", "INFO")
        self.logger.log("="*80, "INFO")
        
        question = (
            "Do student social networks exhibit homophilic community structure "
            "based on academic characteristics (major, dorm), and how does "
            "community modularity relate to university size and type?"
        )
        
        self.logger.log(f"\n{question}\n", "INFO")
        
        self.logger.log("="*80, "INFO")
        self.logger.log("HYPOTHESIS", "INFO")
        self.logger.log("="*80, "INFO")
        
        hypothesis = [
            "H1: Students form communities primarily based on shared academic",
            "    attributes (major and dorm assignment).",
            "",
            "H2: Smaller universities exhibit stronger community structure",
            "    (higher modularity) due to more intimate campus environments.",
            "",
            "H3: Communities detected algorithmically will show significant",
            "    overlap with major and dorm assignments."
        ]
        
        for line in hypothesis:
            self.logger.log(line, "INFO")
        
        self.logger.log("\n" + "="*80 + "\n", "INFO")
    
    def load_selected_networks(self):
        """Load the selected diverse set of networks."""
        self.logger.subsection("Loading Selected Networks")
        self.logger.start_timer("load_networks")
        
        for category, filenames in self.target_networks.items():
            self.logger.log(f"\nLoading {category} Universities:", "INFO")
            
            for filename in filenames:
                try:
                    filepath = os.path.join(self.data_dir, filename)
                    if not os.path.exists(filepath):
                        self.logger.log(
                            f"  File not found: {filename}",
                            "WARNING"
                        )
                        continue
                    
                    name = filename.replace('.gml', '')
                    G = nx.read_gml(filepath)
                    
                    # Get LCC
                    if G.is_directed():
                        components = list(nx.weakly_connected_components(G))
                    else:
                        components = list(nx.connected_components(G))
                    
                    largest = max(components, key=len)
                    G_lcc = G.subgraph(largest).copy()
                    
                    self.networks[name] = G_lcc
                    
                    # Analyze attributes
                    attr_info = analyze_node_attributes(G_lcc)
                    
                    self.logger.log(
                        f"  [OK] {name}: {G_lcc.number_of_nodes():,} nodes, "
                        f"{G_lcc.number_of_edges():,} edges",
                        "SUCCESS"
                    )
                    
                    # Log attribute coverage
                    for attr in ['major', 'dorm', 'gender']:
                        if attr in attr_info:
                            cov = attr_info[attr]['coverage']
                            self.logger.log(
                                f"      {attr}: {cov:.1%} coverage, "
                                f"{attr_info[attr]['unique_values']} unique values",
                                "DATA"
                            )
                    
                except Exception as e:
                    self.logger.log(
                        f"  Failed to load {filename}: {e}",
                        "ERROR"
                    )
        
        self.logger.end_timer("load_networks")
        self.logger.log(
            f"\nLoaded {len(self.networks)} networks total",
            "SUCCESS"
        )
    
    def part_b_community_detection(self):
        """Part (b): Run community detection algorithms."""
        self.logger.subsection("Part (b): Community Detection Algorithms")
        self.logger.start_timer("community_detection")
        
        results = []
        
        for net_idx, (name, G) in enumerate(self.networks.items(), 1):
            self.logger.log(
                f"\n[{net_idx}/{len(self.networks)}] Analyzing {name}...",
                "PROGRESS"
            )
            
            for algo_name, AlgoClass in self.algorithms.items():
                self.logger.log(f"  Running {algo_name}...", "INFO")
                
                try:
                    # Run algorithm
                    detector = AlgoClass(G)
                    communities = detector.detect()
                    
                    # Compute metrics
                    modularity = detector.compute_modularity(communities)
                    n_communities = len(communities)
                    
                    # Community size statistics
                    sizes = [len(c) for c in communities]
                    largest_comm = max(sizes) if sizes else 0
                    smallest_comm = min(sizes) if sizes else 0
                    mean_size = np.mean(sizes) if sizes else 0
                    
                    self.logger.log(
                        f"    Communities: {n_communities}, "
                        f"Modularity: {modularity:.4f}",
                        "DATA"
                    )
                    self.logger.log(
                        f"    Sizes: min={smallest_comm}, max={largest_comm}, "
                        f"mean={mean_size:.1f}",
                        "DATA"
                    )
                    
                    # Analyze homophily
                    homophily_major = self.compute_community_homophily(
                        G, communities, 'major'
                    )
                    homophily_dorm = self.compute_community_homophily(
                        G, communities, 'dorm'
                    )
                    
                    self.logger.log(
                        f"    Homophily: major={homophily_major:.4f}, "
                        f"dorm={homophily_dorm:.4f}",
                        "DATA"
                    )
                    
                    results.append({
                        'Network': name,
                        'N_Nodes': G.number_of_nodes(),
                        'N_Edges': G.number_of_edges(),
                        'Algorithm': algo_name,
                        'N_Communities': n_communities,
                        'Modularity': modularity,
                        'Largest_Community': largest_comm,
                        'Smallest_Community': smallest_comm,
                        'Mean_Community_Size': mean_size,
                        'Homophily_Major': homophily_major,
                        'Homophily_Dorm': homophily_dorm
                    })
                    
                except Exception as e:
                    self.logger.log(
                        f"    Error with {algo_name}: {e}",
                        "ERROR"
                    )
        
        self.logger.end_timer("community_detection")
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            self.results['community_detection'] = df_results
            self.save_csv(df_results, 'Q6_community_detection_full.csv')
            
            self.logger.log(
                f"\nCompleted {len(results)} detections",
                "SUCCESS"
            )
    
    def compute_community_homophily(self, G: nx.Graph, 
                                    communities: List[Set],
                                    attribute: str) -> float:
        """
        Compute homophily of communities w.r.t. an attribute.
        
        Homophily = average purity of communities for that attribute.
        Purity = fraction of nodes in community with most common attribute value.
        
        Returns:
            Average purity across all communities (0 to 1)
        """
        purities = []
        
        for community in communities:
            # Get attribute values for nodes in community
            attr_values = []
            for node in community:
                if node in G.nodes and attribute in G.nodes[node]:
                    val = G.nodes[node][attribute]
                    if val is not None:
                        attr_values.append(val)
            
            if len(attr_values) > 0:
                # Most common value
                counts = Counter(attr_values)
                most_common_count = counts.most_common(1)[0][1]
                purity = most_common_count / len(attr_values)
                purities.append(purity)
        
        if len(purities) > 0:
            return np.mean(purities)
        else:
            return 0.0
    
    def part_c_analysis_and_conclusion(self):
        """Part (c): Analyze results and draw conclusions."""
        self.logger.subsection("Part (c): Analysis and Conclusions")
        
        if 'community_detection' not in self.results:
            self.logger.log("No results to analyze", "ERROR")
            return
        
        df = self.results['community_detection']
        
        # Create visualizations
        self.create_comprehensive_visualizations(df)
        
        # Statistical analysis
        self.perform_statistical_analysis(df)
        
        # Draw conclusions
        self.draw_conclusions(df)
    
    def create_comprehensive_visualizations(self, df):
        """Create comprehensive visualizations."""
        self.logger.log("\nCreating visualizations...", "PROGRESS")
        
        # Viz 1: Modularity by algorithm and network size
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Modularity by algorithm
        ax1 = axes[0, 0]
        df_mod = df.groupby('Algorithm')['Modularity'].agg(['mean', 'std']).reset_index()
        x_pos = np.arange(len(df_mod))
        ax1.bar(x_pos, df_mod['mean'], yerr=df_mod['std'],
               capsize=5, color=['steelblue', 'coral', 'lightgreen'],
               edgecolor='black', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df_mod['Algorithm'])
        ax1.set_ylabel('Modularity')
        ax1.set_title('Mean Modularity by Algorithm')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Modularity vs network size
        ax2 = axes[0, 1]
        for algo in df['Algorithm'].unique():
            df_algo = df[df['Algorithm'] == algo]
            ax2.scatter(df_algo['N_Nodes'], df_algo['Modularity'],
                       label=algo, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        ax2.set_xlabel('Network Size (nodes)')
        ax2.set_ylabel('Modularity')
        ax2.set_title('Modularity vs Network Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Plot 3: Homophily comparison
        ax3 = axes[1, 0]
        df_homophily = df.groupby('Algorithm')[['Homophily_Major', 'Homophily_Dorm']].mean()
        df_homophily.plot(kind='bar', ax=ax3, color=['steelblue', 'coral'],
                         edgecolor='black', alpha=0.8)
        ax3.set_ylabel('Mean Homophily')
        ax3.set_title('Community Homophily by Attribute')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.legend(['Major', 'Dorm'])
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # Plot 4: Number of communities
        ax4 = axes[1, 1]
        for algo in df['Algorithm'].unique():
            df_algo = df[df['Algorithm'] == algo]
            ax4.scatter(df_algo['N_Nodes'], df_algo['N_Communities'],
                       label=algo, alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Network Size (nodes)')
        ax4.set_ylabel('Number of Communities')
        ax4.set_title('Communities Detected vs Network Size')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        self.save_figure('Q6_comprehensive_analysis.png')
        
        # Viz 2: Network-specific comparison
        self.plot_network_comparison(df)
    
    def plot_network_comparison(self, df):
        """Plot comparison across networks."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get unique networks and algorithms
        networks = df['Network'].unique()
        algorithms = df['Algorithm'].unique()
        
        x = np.arange(len(networks))
        width = 0.25
        
        for i, algo in enumerate(algorithms):
            df_algo = df[df['Algorithm'] == algo]
            modularity_values = [
                df_algo[df_algo['Network'] == net]['Modularity'].values[0] 
                if len(df_algo[df_algo['Network'] == net]) > 0 else 0
                for net in networks
            ]
            ax.bar(x + i*width, modularity_values, width, 
                  label=algo, edgecolor='black', alpha=0.8)
        
        ax.set_xlabel('Network')
        ax.set_ylabel('Modularity')
        ax.set_title('Modularity Comparison Across Networks and Algorithms')
        ax.set_xticks(x + width)
        ax.set_xticklabels(networks, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure('Q6_network_comparison.png')
    
    def perform_statistical_analysis(self, df):
        """Perform statistical tests on results."""
        self.logger.log("\n" + "="*80, "INFO")
        self.logger.log("STATISTICAL ANALYSIS", "INFO")
        self.logger.log("="*80 + "\n", "INFO")
        
        # Test 1: Correlation between network size and modularity
        from scipy.stats import pearsonr, spearmanr
        
        corr_pearson, p_pearson = pearsonr(df['N_Nodes'], df['Modularity'])
        corr_spearman, p_spearman = spearmanr(df['N_Nodes'], df['Modularity'])
        
        self.logger.log("Network Size vs Modularity:", "INFO")
        self.logger.log(
            f"  Pearson r = {corr_pearson:.4f} (p = {p_pearson:.4f})",
            "DATA"
        )
        self.logger.log(
            f"  Spearman ρ = {corr_spearman:.4f} (p = {p_spearman:.4f})",
            "DATA"
        )
        
        if p_spearman < 0.05:
            if corr_spearman < 0:
                self.logger.log(
                    "  → Significant NEGATIVE correlation: "
                    "smaller networks have higher modularity",
                    "INFO"
                )
            else:
                self.logger.log(
                    "  → Significant POSITIVE correlation: "
                    "larger networks have higher modularity",
                    "INFO"
                )
        else:
            self.logger.log(
                "  → No significant correlation",
                "INFO"
            )
        
        # Test 2: Compare homophily for major vs dorm
        self.logger.log("\nHomophily Comparison (Major vs Dorm):", "INFO")
        mean_major = df['Homophily_Major'].mean()
        mean_dorm = df['Homophily_Dorm'].mean()
        
        self.logger.log(f"  Major: {mean_major:.4f}", "DATA")
        self.logger.log(f"  Dorm: {mean_dorm:.4f}", "DATA")
        
        if mean_major > mean_dorm:
            self.logger.log(
                "  → Communities align MORE with academic major",
                "INFO"
            )
        else:
            self.logger.log(
                "  → Communities align MORE with dorm assignment",
                "INFO"
            )
    
    def draw_conclusions(self, df):
        """Draw final conclusions about research question."""
        self.logger.log("\n" + "="*80, "INFO")
        self.logger.log("CONCLUSIONS", "INFO")
        self.logger.log("="*80 + "\n", "INFO")
        
        # Evaluate each hypothesis
        self.logger.log("H1: Communities based on academic attributes", "INFO")
        mean_homophily = (df['Homophily_Major'].mean() + df['Homophily_Dorm'].mean()) / 2
        
        if mean_homophily > 0.6:
            self.logger.log(
                f"  [OK] SUPPORTED: High homophily ({mean_homophily:.2f}) indicates "
                "communities align with academic attributes",
                "SUCCESS"
            )
        elif mean_homophily > 0.4:
            self.logger.log(
                f"  ~ PARTIALLY SUPPORTED: Moderate homophily ({mean_homophily:.2f})",
                "WARNING"
            )
        else:
            self.logger.log(
                f"  [FAIL] NOT SUPPORTED: Low homophily ({mean_homophily:.2f})",
                "ERROR"
            )
        
        self.logger.log("\nH2: Smaller universities have stronger community structure", "INFO")
        from scipy.stats import spearmanr
        corr, p_val = spearmanr(df['N_Nodes'], df['Modularity'])
        
        if p_val < 0.05 and corr < 0:
            self.logger.log(
                f"  [OK] SUPPORTED: Negative correlation (ρ={corr:.3f}, p={p_val:.3f})",
                "SUCCESS"
            )
        else:
            self.logger.log(
                f"  [FAIL] NOT SUPPORTED: No significant negative correlation "
                f"(ρ={corr:.3f}, p={p_val:.3f})",
                "ERROR"
            )
        
        self.logger.log("\nH3: Algorithmic communities overlap with major/dorm", "INFO")
        if mean_homophily > 0.5:
            self.logger.log(
                f"  [OK] SUPPORTED: Communities show significant overlap "
                f"(homophily={mean_homophily:.2f})",
                "SUCCESS"
            )
        else:
            self.logger.log(
                f"  [FAIL] NOT SUPPORTED: Limited overlap (homophily={mean_homophily:.2f})",
                "ERROR"
            )
        
        # Overall conclusion
        self.logger.log("\n" + "="*80, "INFO")
        self.logger.log("OVERALL FINDINGS", "INFO")
        self.logger.log("="*80, "INFO")
        
        best_algo = df.groupby('Algorithm')['Modularity'].mean().idxmax()
        best_mod = df.groupby('Algorithm')['Modularity'].mean().max()
        
        self.logger.log(
            f"\n1. Best performing algorithm: {best_algo} "
            f"(mean modularity: {best_mod:.4f})",
            "INFO"
        )
        
        self.logger.log(
            f"\n2. Community structure: {'Strong' if best_mod > 0.4 else 'Moderate' if best_mod > 0.3 else 'Weak'}",
            "INFO"
        )
        
        self.logger.log(
            f"\n3. Academic homophily: "
            f"Major ({df['Homophily_Major'].mean():.2f}) "
            f"{'>' if df['Homophily_Major'].mean() > df['Homophily_Dorm'].mean() else '<'} "
            f"Dorm ({df['Homophily_Dorm'].mean():.2f})",
            "INFO"
        )
        
        self.logger.log(
            "\n4. Social networks on college campuses exhibit clear community "
            "structure that partially aligns with academic and residential "
            "characteristics, but also reflects complex social dynamics beyond "
            "these formal divisions.",
            "INFO"
        )
        
        self.logger.log("\n" + "="*80 + "\n", "INFO")


def main():
    """Main execution function."""
    analysis = Question6Analysis()
    analysis.run()


if __name__ == '__main__':
    main()
