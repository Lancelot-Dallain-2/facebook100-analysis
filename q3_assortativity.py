"""
Question 3: Assortativity Analysis with Facebook100 Dataset

Computes assortativity coefficients for five vertex attributes across
all (or as many as possible) of the FB100 networks:
- (i) student/faculty status
- (ii) major
- (iii) vertex degree  
- (iiii) dorm
- (iiiii) gender

Creates scatter plots and histograms showing assortativity patterns.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.base import NetworkAnalysisBase
from lib.utils import compute_assortativity, analyze_node_attributes, list_available_networks


class Question3Analysis(NetworkAnalysisBase):
    """
    Comprehensive assortativity analysis across all FB100 networks.
    """
    
    def __init__(self, data_dir="fb100/data", output_dir="results/q3"):
        super().__init__(data_dir, output_dir)
        
        # Attributes to analyze
        self.attributes = ['status', 'major', 'degree', 'dorm', 'gender']
        self.assortativity_results = []
    
    def run(self):
        """Execute Question 3 analysis."""
        self.logger.section("QUESTION 3: ASSORTATIVITY ANALYSIS", "=")
        
        # Load all available networks
        self.load_all_networks()
        
        # Compute assortativity for all networks and attributes
        self.compute_all_assortativity()
        
        # Create visualizations
        self.part_a_visualizations()
        
        # Part b: Analyze across all networks
        self.part_b_comprehensive_analysis()
        
        # Summary
        self.print_summary()
        
        self.logger.section("QUESTION 3 COMPLETED", "=")
    
    def load_all_networks(self):
        """Load all available FB100 networks."""
        self.logger.subsection("Loading All FB100 Networks")
        self.logger.start_timer("load_networks")
        
        # Get list of all .gml files
        available_files = list_available_networks(self.data_dir)
        
        if not available_files:
            self.logger.log(f"No networks found in {self.data_dir}", "ERROR")
            return
        
        self.logger.log(f"Found {len(available_files)} network files", "INFO")
        
        # Load networks with progress tracking
        loaded_count = 0
        failed_count = 0
        
        for idx, filename in enumerate(available_files, 1):
            if idx % 10 == 0 or idx == len(available_files):
                self.logger.progress(idx, len(available_files), "networks")
            
            try:
                # Extract network name from filename
                name = filename.replace('.gml', '')
                
                # Load network
                G = self.load_network(filename, name)
                
                # Use largest connected component
                G_lcc = self.get_largest_component(G)
                self.networks[name] = G_lcc
                
                loaded_count += 1
                
            except Exception as e:
                self.logger.log(f"Failed to load {filename}: {e}", "WARNING")
                failed_count += 1
                continue
        
        self.logger.end_timer("load_networks")
        self.logger.log(
            f"Successfully loaded {loaded_count} networks "
            f"({failed_count} failed)",
            "SUCCESS"
        )
    
    def compute_all_assortativity(self):
        """Compute assortativity for all attributes across all networks."""
        self.logger.subsection("Computing Assortativity Coefficients")
        self.logger.start_timer("compute_assortativity")
        
        total_computations = len(self.networks) * len(self.attributes)
        computed = 0
        
        for net_idx, (name, G) in enumerate(self.networks.items(), 1):
            self.logger.log(f"[{net_idx}/{len(self.networks)}] Analyzing {name}...")
            
            # Analyze available attributes
            attr_info = analyze_node_attributes(G)
            
            result_row = {
                'network': name,
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges()
            }
            
            # Compute assortativity for each attribute
            for attr in self.attributes:
                computed += 1
                
                if attr == 'degree':
                    # Special case: degree assortativity
                    try:
                        assort = nx.degree_assortativity_coefficient(G)
                        result_row[f'{attr}_assortativity'] = assort
                        result_row[f'{attr}_coverage'] = 1.0
                        self.logger.log(f"  • {attr}: {assort:.4f}", "DATA")
                    except Exception as e:
                        result_row[f'{attr}_assortativity'] = np.nan
                        result_row[f'{attr}_coverage'] = 0.0
                        self.logger.log(f"  • {attr}: failed ({e})", "WARNING")
                else:
                    # Categorical attributes
                    if attr in attr_info:
                        coverage = attr_info[attr]['coverage']
                        result_row[f'{attr}_coverage'] = coverage
                        
                        if coverage >= 0.5:  # Only compute if sufficient coverage
                            try:
                                assort = nx.attribute_assortativity_coefficient(G, attr)
                                result_row[f'{attr}_assortativity'] = assort
                                self.logger.log(
                                    f"  • {attr}: {assort:.4f} "
                                    f"(coverage: {coverage:.1%})",
                                    "DATA"
                                )
                            except Exception as e:
                                result_row[f'{attr}_assortativity'] = np.nan
                                self.logger.log(
                                    f"  • {attr}: failed ({e})",
                                    "WARNING"
                                )
                        else:
                            result_row[f'{attr}_assortativity'] = np.nan
                            self.logger.log(
                                f"  • {attr}: skipped (low coverage: {coverage:.1%})",
                                "WARNING"
                            )
                    else:
                        result_row[f'{attr}_assortativity'] = np.nan
                        result_row[f'{attr}_coverage'] = 0.0
                        self.logger.log(f"  • {attr}: not available", "WARNING")
            
            self.assortativity_results.append(result_row)
        
        self.logger.end_timer("compute_assortativity")
        
        # Save results to DataFrame
        self.results['assortativity'] = pd.DataFrame(self.assortativity_results)
        self.save_csv(self.results['assortativity'], 'Q3_assortativity_full.csv')
    
    def part_a_visualizations(self):
        """
        Part (a): Create scatter plots and histograms for each attribute.
        """
        self.logger.subsection("Part (a): Creating Visualizations")
        self.logger.start_timer("part_a_viz")
        
        df = self.results['assortativity']
        
        # For each attribute, create a subplot with scatter + histogram
        n_attrs = len(self.attributes)
        fig, axes = plt.subplots(n_attrs, 2, figsize=(14, 4 * n_attrs))
        
        if n_attrs == 1:
            axes = axes.reshape(1, -1)
        
        for idx, attr in enumerate(self.attributes):
            self.logger.log(f"Creating visualizations for {attr}...")
            
            assort_col = f'{attr}_assortativity'
            
            # Filter out NaN values
            df_valid = df[df[assort_col].notna()].copy()
            
            if len(df_valid) == 0:
                self.logger.log(f"  No valid data for {attr}", "WARNING")
                axes[idx, 0].text(0.5, 0.5, f'No data for {attr}',
                                 ha='center', va='center',
                                 transform=axes[idx, 0].transAxes)
                axes[idx, 1].text(0.5, 0.5, f'No data for {attr}',
                                 ha='center', va='center',
                                 transform=axes[idx, 1].transAxes)
                continue
            
            # Scatter plot: assortativity vs network size
            ax_scatter = axes[idx, 0]
            ax_scatter.scatter(df_valid['n_nodes'], df_valid[assort_col],
                             alpha=0.6, s=60, edgecolors='black', linewidth=0.5,
                             color='steelblue')
            ax_scatter.set_xscale('log')
            ax_scatter.axhline(y=0, color='red', linestyle='--', 
                             linewidth=1, alpha=0.7, label='No assortativity')
            ax_scatter.set_xlabel('Network Size (nodes, log scale)')
            ax_scatter.set_ylabel(f'{attr.capitalize()} Assortativity')
            ax_scatter.set_title(f'{attr.capitalize()}: Assortativity vs Network Size')
            ax_scatter.grid(True, alpha=0.3)
            ax_scatter.legend()
            
            # Add statistics
            mean_assort = df_valid[assort_col].mean()
            median_assort = df_valid[assort_col].median()
            stats_text = (f'n={len(df_valid)}\n'
                         f'mean={mean_assort:.4f}\n'
                         f'median={median_assort:.4f}')
            ax_scatter.text(0.05, 0.95, stats_text,
                          transform=ax_scatter.transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                          fontsize=9)
            
            # Histogram: distribution of assortativity values
            ax_hist = axes[idx, 1]
            ax_hist.hist(df_valid[assort_col], bins=30, edgecolor='black',
                        alpha=0.7, color='coral')
            ax_hist.axvline(x=0, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label='No assortativity')
            ax_hist.axvline(x=mean_assort, color='blue', linestyle='-',
                          linewidth=2, alpha=0.7, label=f'Mean={mean_assort:.3f}')
            ax_hist.set_xlabel(f'{attr.capitalize()} Assortativity')
            ax_hist.set_ylabel('Frequency')
            ax_hist.set_title(f'{attr.capitalize()}: Distribution of Assortativity')
            ax_hist.grid(axis='y', alpha=0.3)
            ax_hist.legend()
            
            self.logger.log(
                f"  {attr}: {len(df_valid)} networks, "
                f"mean={mean_assort:.4f}, range=[{df_valid[assort_col].min():.4f}, "
                f"{df_valid[assort_col].max():.4f}]",
                "DATA"
            )
        
        plt.tight_layout()
        self.save_figure('Q3a_assortativity_patterns.png')
        
        self.logger.end_timer("part_a_viz")
    
    def part_b_comprehensive_analysis(self):
        """
        Part (b): Comprehensive analysis across all networks.
        """
        self.logger.subsection("Part (b): Comprehensive Analysis")
        self.logger.start_timer("part_b")
        
        df = self.results['assortativity']
        
        # Create summary statistics for each attribute
        summary_stats = []
        
        for attr in self.attributes:
            assort_col = f'{attr}_assortativity'
            df_valid = df[df[assort_col].notna()]
            
            if len(df_valid) > 0:
                summary_stats.append({
                    'Attribute': attr,
                    'N_Networks': len(df_valid),
                    'Mean': df_valid[assort_col].mean(),
                    'Median': df_valid[assort_col].median(),
                    'Std': df_valid[assort_col].std(),
                    'Min': df_valid[assort_col].min(),
                    'Max': df_valid[assort_col].max(),
                    'Positive_%': (df_valid[assort_col] > 0).sum() / len(df_valid) * 100
                })
        
        df_summary = pd.DataFrame(summary_stats)
        self.save_csv(df_summary, 'Q3b_summary_statistics.csv')
        self.results['summary'] = df_summary
        
        # Create comparative visualization
        self.create_comparative_viz(df_summary)
        
        self.logger.end_timer("part_b")
        
        # Log summary
        self.logger.log("\n[STATS] Summary Statistics by Attribute:", "INFO")
        for _, row in df_summary.iterrows():
            self.logger.log(
                f"  • {row['Attribute']}: mean={row['Mean']:.4f}, "
                f"median={row['Median']:.4f}, "
                f"{row['Positive_%']:.1f}% positive",
                "INFO"
            )
    
    def create_comparative_viz(self, df_summary):
        """Create comparative visualization of all attributes."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Mean assortativity by attribute
        ax1 = axes[0]
        colors = ['steelblue', 'coral', 'lightgreen', 'gold', 'plum']
        bars = ax1.bar(df_summary['Attribute'], df_summary['Mean'],
                      color=colors, edgecolor='black', alpha=0.8)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.set_ylabel('Mean Assortativity Coefficient')
        ax1.set_title('Mean Assortativity by Attribute\n(Across All Networks)')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticklabels(df_summary['Attribute'], rotation=45, ha='right')
        
        # Add error bars (std)
        ax1.errorbar(df_summary['Attribute'], df_summary['Mean'],
                    yerr=df_summary['Std'], fmt='none', 
                    ecolor='black', capsize=5, capthick=2)
        
        # Plot 2: Percentage of networks with positive assortativity
        ax2 = axes[1]
        ax2.bar(df_summary['Attribute'], df_summary['Positive_%'],
               color=colors, edgecolor='black', alpha=0.8)
        ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label='50% threshold')
        ax2.set_ylabel('% Networks with Positive Assortativity')
        ax2.set_title('Prevalence of Positive Assortativity')
        ax2.set_ylim([0, 100])
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_xticklabels(df_summary['Attribute'], rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        self.save_figure('Q3b_comparative_analysis.png')
    
    def print_summary(self):
        """Print comprehensive summary."""
        self.logger.subsection("COMPREHENSIVE SUMMARY")
        
        self.logger.log(f"\n[STATS] Analyzed {len(self.networks)} networks", "INFO")
        
        if 'summary' in self.results:
            df = self.results['summary']
            
            self.logger.log("\n[STATS] KEY FINDINGS:", "INFO")
            
            # Find attribute with highest/lowest mean assortativity
            max_row = df.loc[df['Mean'].idxmax()]
            min_row = df.loc[df['Mean'].idxmin()]
            
            self.logger.log(
                f"  • Highest mean assortativity: {max_row['Attribute']} "
                f"({max_row['Mean']:.4f})",
                "INFO"
            )
            self.logger.log(
                f"  • Lowest mean assortativity: {min_row['Attribute']} "
                f"({min_row['Mean']:.4f})",
                "INFO"
            )
            
            # Interpretation
            self.logger.log("\n[ANALYSIS] INTERPRETATION:", "INFO")
            for _, row in df.iterrows():
                attr = row['Attribute']
                mean = row['Mean']
                
                if mean > 0.1:
                    strength = "Strong"
                elif mean > 0.05:
                    strength = "Moderate"
                elif mean > 0:
                    strength = "Weak"
                else:
                    strength = "Negative/None"
                
                self.logger.log(
                    f"  • {attr}: {strength} assortativity "
                    f"(similar nodes {'tend' if mean > 0 else 'do not tend'} to connect)",
                    "INFO"
                )


def main():
    """Main execution function."""
    analysis = Question3Analysis()
    analysis.run()


if __name__ == '__main__':
    main()
