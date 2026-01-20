"""
Question 5: Label Propagation for Missing Attribute Recovery

Implements label propagation algorithm to recover missing node attributes.

Tests on >10 networks with 10%, 20%, and 30% of attributes randomly removed.
Measures accuracy of attribute recovery.

Note: We implement the SEMI-SUPERVISED label propagation (not community detection).
This is for recovering missing node labels based on network structure + known labels.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Set
import random
from collections import Counter, defaultdict

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.base import NetworkAnalysisBase
from lib.utils import list_available_networks, analyze_node_attributes


class LabelPropagation:
    """
    Semi-supervised label propagation algorithm for attribute recovery.
    
    Algorithm:
    1. Start with known labels
    2. Iteratively propagate labels through edges
    3. Each node adopts most common label among neighbors
    4. Repeat until convergence or max iterations
    """
    
    def __init__(self, G: nx.Graph, attribute: str, max_iter: int = 100):
        """
        Initialize label propagation.
        
        Args:
            G: NetworkX graph
            attribute: Node attribute to propagate
            max_iter: Maximum number of iterations
        """
        self.G = G
        self.attribute = attribute
        self.max_iter = max_iter
        self.labels = {}
        self.known_nodes = set()
        self.unknown_nodes = set()
    
    def fit(self, known_labels: Dict):
        """
        Run label propagation to predict unknown labels.
        
        Args:
            known_labels: Dict mapping node -> label for known nodes
        
        Returns:
            Dict mapping all nodes -> predicted label
        """
        # Initialize labels
        self.labels = known_labels.copy()
        self.known_nodes = set(known_labels.keys())
        self.unknown_nodes = set(self.G.nodes()) - self.known_nodes
        
        # Initialize unknown nodes randomly (from known label set)
        known_label_values = list(set(known_labels.values()))
        for node in self.unknown_nodes:
            self.labels[node] = random.choice(known_label_values)
        
        # Iterative propagation
        for iteration in range(self.max_iter):
            converged = self.propagate_step()
            if converged:
                break
        
        return self.labels
    
    def propagate_step(self) -> bool:
        """
        Perform one iteration of label propagation.
        
        Returns:
            True if converged (no changes), False otherwise
        """
        new_labels = self.labels.copy()
        changed = False
        
        # Only update unknown nodes
        for node in self.unknown_nodes:
            # Get neighbor labels
            neighbor_labels = []
            for neighbor in self.G.neighbors(node):
                if neighbor in self.labels:
                    neighbor_labels.append(self.labels[neighbor])
            
            if neighbor_labels:
                # Most common label among neighbors
                label_counts = Counter(neighbor_labels)
                most_common_label = label_counts.most_common(1)[0][0]
                
                if new_labels[node] != most_common_label:
                    new_labels[node] = most_common_label
                    changed = True
        
        self.labels = new_labels
        return not changed
    
    def predict(self, nodes: List) -> List:
        """
        Get predicted labels for specific nodes.
        
        Args:
            nodes: List of nodes to predict
        
        Returns:
            List of predicted labels
        """
        return [self.labels.get(node, None) for node in nodes]


class Question5Analysis(NetworkAnalysisBase):
    """
    Label propagation evaluation on FB100 networks.
    """
    
    def __init__(self, data_dir="fb100/data", output_dir="results/q5"):
        super().__init__(data_dir, output_dir)
        
        # Evaluation parameters
        self.removal_fractions = [0.1, 0.2, 0.3]
        self.min_network_size = 500
        self.target_networks = 15
        self.n_trials = 3  # Multiple trials for robustness
        
        # Attribute to test (prioritize those with good coverage)
        self.test_attributes = ['major', 'dorm', 'gender', 'status']
    
    def run(self):
        """Execute Question 5 analysis."""
        self.logger.section("QUESTION 5: LABEL PROPAGATION", "=")
        
        # Load suitable networks
        self.load_suitable_networks()
        
        # Evaluate label propagation
        self.evaluate_label_propagation()
        
        # Create visualizations
        self.create_visualizations()
        
        # Summary
        self.print_summary()
        
        self.logger.section("QUESTION 5 COMPLETED", "=")
    
    def load_suitable_networks(self):
        """Load networks with good attribute coverage."""
        self.logger.subsection("Loading Suitable Networks")
        self.logger.start_timer("load_networks")
        
        available_files = list_available_networks(self.data_dir)
        
        if not available_files:
            self.logger.log(f"No networks found in {self.data_dir}", "ERROR")
            return
        
        random.seed(42)
        random.shuffle(available_files)
        
        loaded_count = 0
        
        for idx, filename in enumerate(available_files, 1):
            if loaded_count >= self.target_networks:
                break
            
            try:
                name = filename.replace('.gml', '')
                G = nx.read_gml(os.path.join(self.data_dir, filename))
                
                # Get LCC
                if G.is_directed():
                    components = list(nx.weakly_connected_components(G))
                else:
                    components = list(nx.connected_components(G))
                
                largest = max(components, key=len)
                G_lcc = G.subgraph(largest).copy()
                
                # Check size
                if G_lcc.number_of_nodes() < self.min_network_size:
                    continue
                
                # Check attribute coverage
                attr_analysis = analyze_node_attributes(G_lcc)
                has_good_attr = False
                
                for attr in self.test_attributes:
                    if attr in attr_analysis and attr_analysis[attr]['coverage'] >= 0.8:
                        has_good_attr = True
                        break
                
                if has_good_attr:
                    self.networks[name] = G_lcc
                    loaded_count += 1
                    
                    self.logger.log(
                        f"[{loaded_count}/{self.target_networks}] Loaded {name}: "
                        f"{G_lcc.number_of_nodes():,} nodes",
                        "SUCCESS"
                    )
                    
                    # Log attribute info
                    for attr in self.test_attributes:
                        if attr in attr_analysis:
                            cov = attr_analysis[attr]['coverage']
                            self.logger.log(
                                f"    {attr}: {cov:.1%} coverage",
                                "DATA"
                            )
                            
            except Exception as e:
                self.logger.log(f"Failed to load {filename}: {e}", "WARNING")
                continue
        
        self.logger.end_timer("load_networks")
        self.logger.log(
            f"Loaded {len(self.networks)} networks",
            "SUCCESS"
        )
    
    def evaluate_label_propagation(self):
        """
        Main evaluation loop for all networks and parameters.
        """
        self.logger.subsection("Evaluating Label Propagation")
        self.logger.start_timer("evaluation")
        
        results = []
        
        for net_idx, (name, G) in enumerate(self.networks.items(), 1):
            self.logger.log(
                f"\n[{net_idx}/{len(self.networks)}] Evaluating {name}...",
                "PROGRESS"
            )
            
            # Check which attributes are available
            attr_analysis = analyze_node_attributes(G)
            
            for attr in self.test_attributes:
                if attr not in attr_analysis:
                    continue
                
                coverage = attr_analysis[attr]['coverage']
                if coverage < 0.8:
                    self.logger.log(
                        f"  Skipping {attr}: low coverage ({coverage:.1%})",
                        "WARNING"
                    )
                    continue
                
                self.logger.log(f"  Testing attribute: {attr}", "INFO")
                
                # Extract ground truth labels
                ground_truth = {}
                for node in G.nodes():
                    if attr in G.nodes[node] and G.nodes[node][attr] is not None:
                        ground_truth[node] = G.nodes[node][attr]
                
                if len(ground_truth) < 100:
                    self.logger.log(
                        f"    Skipping: too few labeled nodes ({len(ground_truth)})",
                        "WARNING"
                    )
                    continue
                
                self.logger.log(
                    f"    Ground truth: {len(ground_truth)} nodes labeled",
                    "DATA"
                )
                
                # Test different removal fractions
                for fraction in self.removal_fractions:
                    self.logger.log(f"    Fraction removed: {fraction}", "INFO")
                    
                    # Multiple trials
                    trial_accuracies = []
                    
                    for trial in range(self.n_trials):
                        accuracy = self.run_trial(G, ground_truth, fraction, attr)
                        trial_accuracies.append(accuracy)
                    
                    mean_accuracy = np.mean(trial_accuracies)
                    std_accuracy = np.std(trial_accuracies)
                    
                    results.append({
                        'Network': name,
                        'N_Nodes': G.number_of_nodes(),
                        'Attribute': attr,
                        'Fraction_Removed': fraction,
                        'Mean_Accuracy': mean_accuracy,
                        'Std_Accuracy': std_accuracy,
                        'N_Labeled': len(ground_truth),
                        'N_Trials': self.n_trials
                    })
                    
                    self.logger.log(
                        f"      Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}",
                        "DATA"
                    )
        
        self.logger.end_timer("evaluation")
        
        # Save results
        if results:
            df_results = pd.DataFrame(results)
            self.results['label_propagation'] = df_results
            self.save_csv(df_results, 'Q5_label_propagation_full.csv')
            
            self.logger.log(
                f"Completed {len(results)} evaluations",
                "SUCCESS"
            )
        else:
            self.logger.log("No results generated", "ERROR")
    
    def run_trial(self, G: nx.Graph, ground_truth: Dict, 
                  fraction: float, attr: str) -> float:
        """
        Run one trial of label propagation.
        
        Args:
            G: NetworkX graph
            ground_truth: Dict mapping node -> true label
            fraction: Fraction of labels to remove
            attr: Attribute name
        
        Returns:
            Accuracy (fraction of correctly predicted labels)
        """
        # Randomly select nodes to hide
        all_labeled_nodes = list(ground_truth.keys())
        n_hide = int(len(all_labeled_nodes) * fraction)
        
        hidden_nodes = set(random.sample(all_labeled_nodes, n_hide))
        known_nodes = set(all_labeled_nodes) - hidden_nodes
        
        # Create known labels dict
        known_labels = {node: ground_truth[node] for node in known_nodes}
        
        # Run label propagation
        lp = LabelPropagation(G, attr, max_iter=100)
        predicted_labels = lp.fit(known_labels)
        
        # Evaluate accuracy on hidden nodes
        correct = 0
        total = 0
        
        for node in hidden_nodes:
            if node in predicted_labels:
                predicted = predicted_labels[node]
                true_label = ground_truth[node]
                
                if predicted == true_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        self.logger.subsection("Creating Visualizations")
        self.logger.start_timer("visualization")
        
        if 'label_propagation' not in self.results:
            self.logger.log("No results to visualize", "WARNING")
            return
        
        df = self.results['label_propagation']
        
        # Visualization 1: Accuracy by fraction removed
        self.plot_accuracy_by_fraction(df)
        
        # Visualization 2: Accuracy by attribute
        self.plot_accuracy_by_attribute(df)
        
        # Visualization 3: Network comparison
        self.plot_network_comparison(df)
        
        self.logger.end_timer("visualization")
    
    def plot_accuracy_by_fraction(self, df):
        """Plot accuracy vs fraction removed."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by attribute and fraction
        df_grouped = df.groupby(['Attribute', 'Fraction_Removed'])['Mean_Accuracy'].mean().reset_index()
        
        for attr in df_grouped['Attribute'].unique():
            df_attr = df_grouped[df_grouped['Attribute'] == attr]
            ax.plot(df_attr['Fraction_Removed'], df_attr['Mean_Accuracy'],
                   marker='o', linewidth=2, markersize=8, label=attr)
        
        ax.set_xlabel('Fraction of Labels Removed')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Label Propagation Accuracy vs Fraction Removed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        self.save_figure('Q5_accuracy_by_fraction.png')
    
    def plot_accuracy_by_attribute(self, df):
        """Plot accuracy comparison by attribute."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Average across all fractions and networks
        df_grouped = df.groupby('Attribute')['Mean_Accuracy'].agg(['mean', 'std']).reset_index()
        
        colors = ['steelblue', 'coral', 'lightgreen', 'gold']
        x_pos = np.arange(len(df_grouped))
        
        bars = ax.bar(x_pos, df_grouped['mean'], 
                     yerr=df_grouped['std'],
                     color=colors[:len(df_grouped)],
                     edgecolor='black', alpha=0.8,
                     capsize=5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_grouped['Attribute'])
        ax.set_ylabel('Mean Accuracy')
        ax.set_title('Label Propagation Accuracy by Attribute\n(Averaged across all fractions)')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        self.save_figure('Q5_accuracy_by_attribute.png')
    
    def plot_network_comparison(self, df):
        """Plot accuracy comparison across networks."""
        # Select one attribute and one fraction for clearer comparison
        if len(df) == 0:
            return
        
        # Use most common attribute and middle fraction
        common_attr = df['Attribute'].mode()[0]
        middle_frac = sorted(df['Fraction_Removed'].unique())[len(df['Fraction_Removed'].unique())//2]
        
        df_subset = df[(df['Attribute'] == common_attr) & 
                       (df['Fraction_Removed'] == middle_frac)]
        
        if len(df_subset) == 0:
            return
        
        df_subset = df_subset.sort_values('Mean_Accuracy', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(df_subset))
        bars = ax.bar(x_pos, df_subset['Mean_Accuracy'],
                     yerr=df_subset['Std_Accuracy'],
                     color='steelblue', edgecolor='black', alpha=0.8,
                     capsize=3)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df_subset['Network'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(
            f'Label Propagation Accuracy by Network\n'
            f'(Attribute: {common_attr}, Fraction Removed: {middle_frac})'
        )
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        self.save_figure('Q5_network_comparison.png')
    
    def print_summary(self):
        """Print comprehensive summary."""
        self.logger.subsection("COMPREHENSIVE SUMMARY")
        
        if 'label_propagation' not in self.results:
            self.logger.log("No results to summarize", "WARNING")
            return
        
        df = self.results['label_propagation']
        
        self.logger.log(f"\n[STATS] Evaluated {len(self.networks)} networks", "INFO")
        self.logger.log(f"[STATS] Total evaluations: {len(df)}", "INFO")
        
        # Average accuracy by attribute
        self.logger.log("\n[STATS] AVERAGE ACCURACY BY ATTRIBUTE:", "INFO")
        df_attr = df.groupby('Attribute')['Mean_Accuracy'].mean().sort_values(ascending=False)
        for attr, acc in df_attr.items():
            self.logger.log(f"  • {attr}: {acc:.4f}", "INFO")
        
        # Effect of fraction removed
        self.logger.log("\n[STATS] EFFECT OF FRACTION REMOVED:", "INFO")
        df_frac = df.groupby('Fraction_Removed')['Mean_Accuracy'].mean()
        for frac, acc in df_frac.items():
            self.logger.log(f"  • {frac:.0%} removed: {acc:.4f} accuracy", "INFO")
        
        # Best and worst performers
        best_row = df.loc[df['Mean_Accuracy'].idxmax()]
        worst_row = df.loc[df['Mean_Accuracy'].idxmin()]
        
        self.logger.log("\n[STATS] BEST PERFORMANCE:", "INFO")
        self.logger.log(
            f"  • Network: {best_row['Network']}, "
            f"Attribute: {best_row['Attribute']}, "
            f"Fraction: {best_row['Fraction_Removed']:.0%}",
            "INFO"
        )
        self.logger.log(
            f"  • Accuracy: {best_row['Mean_Accuracy']:.4f} ± "
            f"{best_row['Std_Accuracy']:.4f}",
            "INFO"
        )
        
        self.logger.log("\n[STATS] WORST PERFORMANCE:", "INFO")
        self.logger.log(
            f"  • Network: {worst_row['Network']}, "
            f"Attribute: {worst_row['Attribute']}, "
            f"Fraction: {worst_row['Fraction_Removed']:.0%}",
            "INFO"
        )
        self.logger.log(
            f"  • Accuracy: {worst_row['Mean_Accuracy']:.4f} ± "
            f"{worst_row['Std_Accuracy']:.4f}",
            "INFO"
        )


def main():
    """Main execution function."""
    random.seed(42)
    np.random.seed(42)
    
    analysis = Question5Analysis()
    analysis.run()


if __name__ == '__main__':
    main()
