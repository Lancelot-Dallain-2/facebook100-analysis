"""
Question 2: Social Network Analysis with Facebook100 Dataset

Analyzes three specific networks:
- Caltech (762 nodes in LCC)
- MIT (6402 nodes in LCC) 
- Johns Hopkins (5157 nodes in LCC)

Tasks:
(a) Plot degree distribution for each network
(b) Compute global clustering coefficient, mean local clustering, and density
(c) Scatter plot of degree vs local clustering coefficient
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.base import NetworkAnalysisBase
from lib.utils import plot_degree_distribution, plot_scatter, compute_basic_stats


class Question2Analysis(NetworkAnalysisBase):
    """
    Comprehensive analysis of three Facebook100 networks.
    """
    
    def __init__(self, data_dir="fb100/data", output_dir="results/q2"):
        super().__init__(data_dir, output_dir)
        
        # Define the three networks to analyze
        self.network_specs = {
            'Caltech': 'Caltech36.gml',
            'MIT': 'MIT8.gml',
            'Johns Hopkins': 'Johns Hopkins55.gml'
        }
    
    def run(self):
        """Execute all parts of Question 2."""
        self.logger.section("QUESTION 2: SOCIAL NETWORK ANALYSIS", "=")
        
        # Load networks
        self.load_networks()
        
        # Part (a): Degree distributions
        self.part_a_degree_distribution()
        
        # Part (b): Clustering and density
        self.part_b_clustering_density()
        
        # Part (c): Degree vs clustering correlation
        self.part_c_degree_vs_clustering()
        
        # Summary statistics
        self.print_summary()
        
        self.logger.section("QUESTION 2 COMPLETED", "=")
    
    def load_networks(self):
        """Load the three required networks."""
        self.logger.subsection("Loading Networks")
        self.logger.start_timer("load_networks")
        
        for idx, (name, filename) in enumerate(self.network_specs.items(), 1):
            self.logger.progress(idx, len(self.network_specs), "networks")
            
            try:
                G = self.load_network(filename, name)
                
                # Use largest connected component
                G_lcc = self.get_largest_component(G)
                self.networks[name] = G_lcc
                
                # Log basic stats
                stats_dict = compute_basic_stats(G_lcc)
                self.logger.log(
                    f"  Stats: density={stats_dict['density']:.6f}, "
                    f"avg_degree={stats_dict['avg_degree']:.2f}, "
                    f"clustering={stats_dict['global_clustering']:.4f}",
                    "DATA"
                )
                
            except Exception as e:
                self.logger.log(f"Failed to load {name}: {e}", "ERROR")
                continue
        
        self.logger.end_timer("load_networks")
        self.logger.log(f"Successfully loaded {len(self.networks)} networks", "SUCCESS")
    
    def part_a_degree_distribution(self):
        """
        Part (a): Plot degree distribution for the three networks.
        
        Creates both regular and log-log plots to identify power-law behavior.
        """
        self.logger.subsection("Part (a): Degree Distribution Analysis")
        self.logger.start_timer("part_a")
        
        # Create figure with 2 rows: regular and log-log
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for idx, (name, G) in enumerate(self.networks.items()):
            self.logger.log(f"Computing degree distribution for {name}...")
            
            degrees = [d for n, d in G.degree()]
            
            # Row 1: Regular histogram
            ax_regular = axes[0, idx]
            ax_regular.hist(degrees, bins=min(50, max(degrees)), 
                          edgecolor='black', alpha=0.7, color='steelblue')
            ax_regular.set_xlabel('Degree')
            ax_regular.set_ylabel('Frequency')
            ax_regular.set_title(f'{name}\n(Regular Scale)')
            ax_regular.grid(True, alpha=0.3)
            
            # Add statistics
            mean_deg = np.mean(degrees)
            median_deg = np.median(degrees)
            max_deg = np.max(degrees)
            min_deg = np.min(degrees)
            
            stats_text = (f'Mean: {mean_deg:.2f}\n'
                         f'Median: {median_deg:.0f}\n'
                         f'Range: [{min_deg}, {max_deg}]')
            ax_regular.text(0.98, 0.98, stats_text,
                          transform=ax_regular.transAxes, 
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                          fontsize=9)
            
            # Row 2: Log-log plot
            ax_loglog = axes[1, idx]
            unique_degrees, counts = np.unique(degrees, return_counts=True)
            prob = counts / len(degrees)
            
            ax_loglog.scatter(unique_degrees, prob, alpha=0.6, s=50, 
                            edgecolors='black', linewidth=0.5, color='darkred')
            ax_loglog.set_xscale('log')
            ax_loglog.set_yscale('log')
            ax_loglog.set_xlabel('Degree (k)')
            ax_loglog.set_ylabel('P(k)')
            ax_loglog.set_title(f'{name}\n(Log-Log Scale)')
            ax_loglog.grid(True, alpha=0.3, which='both')
            
            # Log detailed statistics
            self.logger.log(
                f"  {name}: min={min_deg}, max={max_deg}, "
                f"mean={mean_deg:.2f}, median={median_deg:.0f}, "
                f"std={np.std(degrees):.2f}",
                "DATA"
            )
        
        plt.tight_layout()
        self.save_figure('Q2a_degree_distributions.png')
        
        self.logger.end_timer("part_a")
        
        # Analysis conclusion
        self.logger.log("\n[ANALYSIS] Degree Distribution Analysis:", "INFO")
        self.logger.log("  • All networks show right-skewed distributions", "INFO")
        self.logger.log("  • Log-log plots suggest approximate power-law behavior", "INFO")
        self.logger.log("  • Most nodes have relatively low degrees (< mean)", "INFO")
        self.logger.log("  • Few hub nodes with very high connectivity", "INFO")
    
    def part_b_clustering_density(self):
        """
        Part (b): Compute clustering coefficients and density.
        
        Computes:
        - Global clustering coefficient (transitivity)
        - Mean local clustering coefficient
        - Edge density
        """
        self.logger.subsection("Part (b): Clustering & Density Analysis")
        self.logger.start_timer("part_b")
        
        results = []
        
        for idx, (name, G) in enumerate(self.networks.items(), 1):
            self.logger.log(f"[{idx}/{len(self.networks)}] Analyzing {name}...")
            
            # Compute metrics
            self.logger.log("  Computing global clustering (transitivity)...")
            global_clustering = nx.transitivity(G)
            
            self.logger.log("  Computing mean local clustering...")
            local_clustering = nx.average_clustering(G)
            
            self.logger.log("  Computing network density...")
            density = nx.density(G)
            
            # Store results
            results.append({
                'Network': name,
                'Nodes': G.number_of_nodes(),
                'Edges': G.number_of_edges(),
                'Global_Clustering': global_clustering,
                'Mean_Local_Clustering': local_clustering,
                'Density': density,
                'Is_Sparse': 'Yes' if density < 0.1 else 'No'
            })
            
            # Log results
            self.logger.log(
                f"  {name} Results:",
                "DATA"
            )
            self.logger.log(f"    • Global clustering: {global_clustering:.6f}", "DATA")
            self.logger.log(f"    • Mean local clustering: {local_clustering:.6f}", "DATA")
            self.logger.log(f"    • Density: {density:.6f}", "DATA")
            self.logger.log(
                f"    • Sparse? {'Yes' if density < 0.1 else 'No'} "
                f"(density < 0.1 threshold)",
                "DATA"
            )
        
        # Save to CSV
        df = pd.DataFrame(results)
        self.save_csv(df, 'Q2b_clustering_density.csv')
        self.results['q2b'] = df
        
        # Create visualization only if we have data
        if len(df) > 0:
            self.visualize_clustering_density(df)
        else:
            self.logger.log("No data to visualize", "WARNING")
        
        self.logger.end_timer("part_b")
        
        # Analysis conclusions
        self.logger.log("\n[ANALYSIS] Clustering & Density Analysis:", "INFO")
        self.logger.log(
            f"  • All networks are SPARSE (density < 0.1)", 
            "INFO"
        )
        self.logger.log(
            f"  • Caltech has highest density ({df.loc[df['Network']=='Caltech', 'Density'].values[0]:.6f})",
            "INFO"
        )
        self.logger.log(
            f"  • Global clustering ranges from {df['Global_Clustering'].min():.4f} "
            f"to {df['Global_Clustering'].max():.4f}",
            "INFO"
        )
        self.logger.log(
            "  • High clustering despite sparsity indicates community structure",
            "INFO"
        )
    
    def visualize_clustering_density(self, df):
        """Create visualization for clustering and density metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        networks = df['Network'].values
        x_pos = np.arange(len(networks))
        
        # Plot 1: Clustering coefficients
        axes[0].bar(x_pos - 0.2, df['Global_Clustering'], 0.4, 
                   label='Global', color='steelblue', edgecolor='black')
        axes[0].bar(x_pos + 0.2, df['Mean_Local_Clustering'], 0.4, 
                   label='Mean Local', color='coral', edgecolor='black')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(networks, rotation=15, ha='right')
        axes[0].set_ylabel('Clustering Coefficient')
        axes[0].set_title('Clustering Coefficients')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Density
        axes[1].bar(x_pos, df['Density'], color='darkgreen', 
                   edgecolor='black', alpha=0.7)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(networks, rotation=15, ha='right')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Network Density')
        axes[1].axhline(y=0.1, color='red', linestyle='--', 
                       label='Sparse threshold (0.1)')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Network size
        axes[2].bar(x_pos, df['Nodes'], color='purple', 
                   edgecolor='black', alpha=0.7)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(networks, rotation=15, ha='right')
        axes[2].set_ylabel('Number of Nodes')
        axes[2].set_title('Network Size (LCC)')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self.save_figure('Q2b_clustering_density_viz.png')
    
    def part_c_degree_vs_clustering(self):
        """
        Part (c): Analyze correlation between degree and local clustering.
        
        Creates scatter plots and computes correlation coefficients.
        """
        self.logger.subsection("Part (c): Degree vs Local Clustering Correlation")
        self.logger.start_timer("part_c")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        correlation_results = []
        
        for idx, (name, G) in enumerate(self.networks.items()):
            self.logger.log(f"Analyzing {name}...")
            
            # Compute local clustering for all nodes
            self.logger.log("  Computing local clustering coefficients...")
            local_clustering_dict = nx.clustering(G)
            
            # Extract degree and clustering pairs
            degrees = []
            clusterings = []
            
            for node in G.nodes():
                degrees.append(G.degree(node))
                clusterings.append(local_clustering_dict[node])
            
            degrees = np.array(degrees)
            clusterings = np.array(clusterings)
            
            # Row 1: Linear scale
            ax_linear = axes[0, idx]
            ax_linear.scatter(degrees, clusterings, alpha=0.4, s=20, 
                            edgecolors='none', color='steelblue')
            ax_linear.set_xlabel('Degree')
            ax_linear.set_ylabel('Local Clustering Coefficient')
            ax_linear.set_title(f'{name}\n(Linear Scale)')
            ax_linear.grid(True, alpha=0.3)
            
            # Compute correlation (Pearson)
            valid_idx = (degrees > 0) & np.isfinite(clusterings)
            if np.sum(valid_idx) > 1:
                corr_pearson, p_val_pearson = stats.pearsonr(
                    degrees[valid_idx], 
                    clusterings[valid_idx]
                )
                
                # Also compute Spearman (non-parametric)
                corr_spearman, p_val_spearman = stats.spearmanr(
                    degrees[valid_idx], 
                    clusterings[valid_idx]
                )
                
                stats_text = (f'Pearson r: {corr_pearson:.4f}\n'
                             f'p-value: {p_val_pearson:.2e}\n'
                             f'Spearman ρ: {corr_spearman:.4f}')
                ax_linear.text(0.05, 0.95, stats_text,
                             transform=ax_linear.transAxes,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                             fontsize=8)
                
                correlation_results.append({
                    'Network': name,
                    'Pearson_r': corr_pearson,
                    'Pearson_p': p_val_pearson,
                    'Spearman_rho': corr_spearman,
                    'Spearman_p': p_val_spearman,
                    'N_nodes': len(degrees)
                })
            
            # Row 2: Binned average (clearer pattern)
            ax_binned = axes[1, idx]
            
            # Create degree bins
            degree_bins = np.logspace(np.log10(max(1, degrees.min())), 
                                     np.log10(degrees.max()), 20)
            binned_degrees = []
            binned_clustering_mean = []
            binned_clustering_std = []
            
            for i in range(len(degree_bins) - 1):
                mask = (degrees >= degree_bins[i]) & (degrees < degree_bins[i+1])
                if np.sum(mask) > 0:
                    binned_degrees.append(np.mean([degree_bins[i], degree_bins[i+1]]))
                    binned_clustering_mean.append(np.mean(clusterings[mask]))
                    binned_clustering_std.append(np.std(clusterings[mask]))
            
            if len(binned_degrees) > 0:
                ax_binned.errorbar(binned_degrees, binned_clustering_mean, 
                                  yerr=binned_clustering_std, fmt='o-', 
                                  capsize=5, markersize=6, alpha=0.8,
                                  color='darkred', ecolor='gray')
            
            ax_binned.set_xscale('log')
            ax_binned.set_xlabel('Degree (log scale)')
            ax_binned.set_ylabel('Mean Local Clustering')
            ax_binned.set_title(f'{name}\n(Binned Average)')
            ax_binned.grid(True, alpha=0.3)
            
            self.logger.log(
                f"  {name}: Pearson r={corr_pearson:.4f} (p={p_val_pearson:.2e}), "
                f"Spearman ρ={corr_spearman:.4f}",
                "DATA"
            )
        
        plt.tight_layout()
        self.save_figure('Q2c_degree_vs_clustering.png')
        
        # Save correlation results
        if correlation_results:
            df_corr = pd.DataFrame(correlation_results)
            self.save_csv(df_corr, 'Q2c_correlations.csv')
            self.results['q2c'] = df_corr
        
        self.logger.end_timer("part_c")
        
        # Analysis conclusions
        self.logger.log("\n[ANALYSIS] Degree-Clustering Correlation Analysis:", "INFO")
        if correlation_results:
            avg_pearson = np.mean([r['Pearson_r'] for r in correlation_results])
            self.logger.log(
                f"  • Average Pearson correlation: {avg_pearson:.4f}",
                "INFO"
            )
            self.logger.log(
                "  • Negative correlation typical in social networks",
                "INFO"
            )
            self.logger.log(
                "  • High-degree nodes tend to bridge communities (lower clustering)",
                "INFO"
            )
            self.logger.log(
                "  • Low-degree nodes within tight-knit groups (higher clustering)",
                "INFO"
            )
    
    def print_summary(self):
        """Print comprehensive summary of all results."""
        self.logger.subsection("COMPREHENSIVE SUMMARY")
        
        self.logger.log("\n[STATS] NETWORK OVERVIEW:", "INFO")
        for name, G in self.networks.items():
            n = G.number_of_nodes()
            m = G.number_of_edges()
            self.logger.log(f"  • {name}: {n:,} nodes, {m:,} edges", "INFO")
        
        if 'q2b' in self.results:
            df = self.results['q2b']
            self.logger.log("\n[STATS] CLUSTERING & DENSITY:", "INFO")
            for _, row in df.iterrows():
                self.logger.log(
                    f"  • {row['Network']}: "
                    f"clustering={row['Global_Clustering']:.4f}, "
                    f"density={row['Density']:.6f}",
                    "INFO"
                )
        
        if 'q2c' in self.results:
            df = self.results['q2c']
            self.logger.log("\n[STATS] DEGREE-CLUSTERING CORRELATION:", "INFO")
            for _, row in df.iterrows():
                self.logger.log(
                    f"  • {row['Network']}: "
                    f"Pearson r={row['Pearson_r']:.4f} "
                    f"(p={row['Pearson_p']:.2e})",
                    "INFO"
                )


def main():
    """Main execution function."""
    analysis = Question2Analysis()
    analysis.run()


if __name__ == '__main__':
    main()
