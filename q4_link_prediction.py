"""
Question 4: Link Prediction

Implements and evaluates three link prediction metrics:
1. Common Neighbors
2. Jaccard Coefficient
3. Adamic/Adar

Evaluation procedure:
- Remove fraction f of edges (f ∈ [0.05, 0.1, 0.15, 0.2])
- Predict missing edges using each metric
- Evaluate using Precision@k and Recall@k (k ∈ [50, 100, 200])
- Test on >10 graphs from FB100
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Tuple, Set
import random
from collections import defaultdict

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.base import NetworkAnalysisBase
from lib.utils import list_available_networks


class LinkPredictor:
    """
    Base class for link prediction algorithms.
    Following sklearn-like API.
    """
    
    def __init__(self, G: nx.Graph):
        """
        Initialize predictor with a graph.
        
        Args:
            G: NetworkX graph (training graph with removed edges)
        """
        self.G = G
        self.scores = {}
    
    def predict(self, node_pairs: List[Tuple]) -> np.ndarray:
        """
        Predict scores for given node pairs.
        
        Args:
            node_pairs: List of (u, v) tuples
        
        Returns:
            Array of scores (higher = more likely to be an edge)
        """
        raise NotImplementedError
    
    def score_all_non_edges(self) -> dict:
        """
        Compute scores for all non-edges in the graph.
        
        Returns:
            Dictionary mapping (u, v) -> score
        """
        raise NotImplementedError


class CommonNeighborsPredictor(LinkPredictor):
    """
    Common Neighbors: score(u,v) = |Γ(u) ∩ Γ(v)|
    where Γ(x) is the set of neighbors of node x.
    """
    
    def predict(self, node_pairs: List[Tuple]) -> np.ndarray:
        scores = []
        for u, v in node_pairs:
            if u in self.G and v in self.G:
                # Get neighbors
                neighbors_u = set(self.G.neighbors(u))
                neighbors_v = set(self.G.neighbors(v))
                # Count common neighbors
                score = len(neighbors_u & neighbors_v)
            else:
                score = 0
            scores.append(score)
        return np.array(scores)
    
    def score_all_non_edges(self) -> dict:
        scores = {}
        nodes = list(self.G.nodes())
        
        for i, u in enumerate(nodes):
            neighbors_u = set(self.G.neighbors(u))
            for v in nodes[i+1:]:
                if not self.G.has_edge(u, v):
                    neighbors_v = set(self.G.neighbors(v))
                    score = len(neighbors_u & neighbors_v)
                    if score > 0:  # Only store positive scores
                        scores[(u, v)] = score
        
        return scores


class JaccardPredictor(LinkPredictor):
    """
    Jaccard Coefficient: score(u,v) = |Γ(u) ∩ Γ(v)| / |Γ(u) ∪ Γ(v)|
    """
    
    def predict(self, node_pairs: List[Tuple]) -> np.ndarray:
        scores = []
        for u, v in node_pairs:
            if u in self.G and v in self.G:
                neighbors_u = set(self.G.neighbors(u))
                neighbors_v = set(self.G.neighbors(v))
                
                intersection = len(neighbors_u & neighbors_v)
                union = len(neighbors_u | neighbors_v)
                
                if union > 0:
                    score = intersection / union
                else:
                    score = 0
            else:
                score = 0
            scores.append(score)
        return np.array(scores)
    
    def score_all_non_edges(self) -> dict:
        scores = {}
        nodes = list(self.G.nodes())
        
        for i, u in enumerate(nodes):
            neighbors_u = set(self.G.neighbors(u))
            for v in nodes[i+1:]:
                if not self.G.has_edge(u, v):
                    neighbors_v = set(self.G.neighbors(v))
                    
                    intersection = len(neighbors_u & neighbors_v)
                    union = len(neighbors_u | neighbors_v)
                    
                    if union > 0:
                        score = intersection / union
                        if score > 0:
                            scores[(u, v)] = score
        
        return scores


class AdamicAdarPredictor(LinkPredictor):
    """
    Adamic/Adar: score(u,v) = Σ_{z ∈ Γ(u) ∩ Γ(v)} 1 / log(|Γ(z)|)
    
    Weights common neighbors by inverse log of their degree.
    High-degree common neighbors are less valuable than low-degree ones.
    """
    
    def predict(self, node_pairs: List[Tuple]) -> np.ndarray:
        scores = []
        for u, v in node_pairs:
            if u in self.G and v in self.G:
                neighbors_u = set(self.G.neighbors(u))
                neighbors_v = set(self.G.neighbors(v))
                common = neighbors_u & neighbors_v
                
                score = 0
                for z in common:
                    degree_z = self.G.degree(z)
                    if degree_z > 1:  # Avoid log(1) = 0 and log(0) = -inf
                        score += 1 / np.log(degree_z)
            else:
                score = 0
            scores.append(score)
        return np.array(scores)
    
    def score_all_non_edges(self) -> dict:
        scores = {}
        nodes = list(self.G.nodes())
        
        for i, u in enumerate(nodes):
            neighbors_u = set(self.G.neighbors(u))
            for v in nodes[i+1:]:
                if not self.G.has_edge(u, v):
                    neighbors_v = set(self.G.neighbors(v))
                    common = neighbors_u & neighbors_v
                    
                    score = 0
                    for z in common:
                        degree_z = self.G.degree(z)
                        if degree_z > 1:
                            score += 1 / np.log(degree_z)
                    
                    if score > 0:
                        scores[(u, v)] = score
        
        return scores


class Question4Analysis(NetworkAnalysisBase):
    """
    Link prediction evaluation on FB100 networks.
    """
    
    def __init__(self, data_dir="fb100/data", output_dir="results/q4"):
        super().__init__(data_dir, output_dir)
        
        # Evaluation parameters
        self.fractions = [0.05, 0.1, 0.15, 0.2]
        self.k_values = [50, 100, 200]
        self.metrics = {
            'common_neighbors': CommonNeighborsPredictor,
            'jaccard': JaccardPredictor,
            'adamic_adar': AdamicAdarPredictor
        }
        
        # Minimum network size (avoid too small networks)
        self.min_network_size = 500
        self.target_networks = 15  # Aim for >10 networks
    
    def run(self):
        """Execute Question 4 analysis."""
        self.logger.section("QUESTION 4: LINK PREDICTION", "=")
        
        # Load suitable networks
        self.load_suitable_networks()
        
        # Run link prediction evaluation
        self.evaluate_link_prediction()
        
        # Create visualizations
        self.create_visualizations()
        
        # Summary
        self.print_summary()
        
        self.logger.section("QUESTION 4 COMPLETED", "=")
    
    def load_suitable_networks(self):
        """Load networks suitable for link prediction (not too small)."""
        self.logger.subsection("Loading Suitable Networks")
        self.logger.start_timer("load_networks")
        
        available_files = list_available_networks(self.data_dir)
        
        if not available_files:
            self.logger.log(f"No networks found in {self.data_dir}", "ERROR")
            return
        
        # Sort by estimated size (from filename convention)
        # Prioritize medium-sized networks
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
                
                # Check if network is suitable
                if G_lcc.number_of_nodes() >= self.min_network_size:
                    self.networks[name] = G_lcc
                    loaded_count += 1
                    self.logger.log(
                        f"[{loaded_count}/{self.target_networks}] Loaded {name}: "
                        f"{G_lcc.number_of_nodes():,} nodes, "
                        f"{G_lcc.number_of_edges():,} edges",
                        "SUCCESS"
                    )
                else:
                    self.logger.log(
                        f"Skipped {name}: too small "
                        f"({G_lcc.number_of_nodes()} < {self.min_network_size})",
                        "WARNING"
                    )
                    
            except Exception as e:
                self.logger.log(f"Failed to load {filename}: {e}", "WARNING")
                continue
        
        self.logger.end_timer("load_networks")
        self.logger.log(
            f"Loaded {len(self.networks)} networks for link prediction",
            "SUCCESS"
        )
    
    def remove_edges(self, G: nx.Graph, fraction: float) -> Tuple[nx.Graph, Set]:
        """
        Remove a fraction of edges from the graph.
        
        Args:
            G: Original graph
            fraction: Fraction of edges to remove (0 < fraction < 1)
        
        Returns:
            (training_graph, removed_edges)
        """
        edges = list(G.edges())
        n_remove = int(len(edges) * fraction)
        
        # Randomly select edges to remove
        random.shuffle(edges)
        removed_edges = set(edges[:n_remove])
        
        # Create training graph
        G_train = G.copy()
        G_train.remove_edges_from(removed_edges)
        
        # Ensure graph is still connected (use LCC if not)
        if not nx.is_connected(G_train):
            largest_cc = max(nx.connected_components(G_train), key=len)
            G_train = G_train.subgraph(largest_cc).copy()
            
            # Filter removed edges to only those in LCC
            removed_edges = {(u, v) for u, v in removed_edges 
                           if u in G_train and v in G_train}
        
        return G_train, removed_edges
    
    def evaluate_predictions(self, scores: dict, removed_edges: Set, 
                            k: int) -> Tuple[float, float]:
        """
        Evaluate link predictions using Precision@k and Recall@k.
        
        Args:
            scores: Dictionary mapping (u, v) -> score
            removed_edges: Set of true edges that were removed
            k: Number of top predictions to evaluate
        
        Returns:
            (precision, recall)
        """
        if len(scores) == 0:
            return 0.0, 0.0
        
        # Sort predictions by score (descending)
        sorted_predictions = sorted(scores.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
        
        # Get top-k predictions
        top_k = sorted_predictions[:k]
        top_k_edges = {edge for edge, score in top_k}
        
        # Convert removed_edges to undirected (both (u,v) and (v,u))
        removed_undirected = set()
        for u, v in removed_edges:
            removed_undirected.add((u, v))
            removed_undirected.add((v, u))
        
        # Count correct predictions
        correct = len(top_k_edges & removed_undirected)
        
        # Precision@k: fraction of top-k that are correct
        precision = correct / k if k > 0 else 0
        
        # Recall@k: fraction of removed edges found in top-k
        recall = correct / len(removed_edges) if len(removed_edges) > 0 else 0
        
        return precision, recall
    
    def evaluate_link_prediction(self):
        """
        Main evaluation loop for all networks, metrics, and parameters.
        """
        self.logger.subsection("Evaluating Link Prediction")
        self.logger.start_timer("evaluation")
        
        results = []
        total_evals = (len(self.networks) * len(self.metrics) * 
                      len(self.fractions) * len(self.k_values))
        current_eval = 0
        
        for net_idx, (name, G) in enumerate(self.networks.items(), 1):
            self.logger.log(
                f"\n[{net_idx}/{len(self.networks)}] Evaluating {name}...",
                "PROGRESS"
            )
            
            for fraction in self.fractions:
                self.logger.log(f"  Fraction removed: {fraction}", "INFO")
                
                # Remove edges
                G_train, removed_edges = self.remove_edges(G, fraction)
                
                self.logger.log(
                    f"    Training graph: {G_train.number_of_nodes()} nodes, "
                    f"{G_train.number_of_edges()} edges",
                    "DATA"
                )
                self.logger.log(
                    f"    Removed edges: {len(removed_edges)}",
                    "DATA"
                )
                
                for metric_name, PredictorClass in self.metrics.items():
                    self.logger.log(f"    Metric: {metric_name}", "INFO")
                    
                    # Create predictor
                    predictor = PredictorClass(G_train)
                    
                    # Compute scores for all non-edges
                    self.logger.log("      Computing scores...", "PROGRESS")
                    scores = predictor.score_all_non_edges()
                    self.logger.log(
                        f"      Computed {len(scores)} scores",
                        "DATA"
                    )
                    
                    for k in self.k_values:
                        current_eval += 1
                        
                        # Evaluate
                        precision, recall = self.evaluate_predictions(
                            scores, removed_edges, k
                        )
                        
                        results.append({
                            'Network': name,
                            'N_Nodes': G.number_of_nodes(),
                            'N_Edges': G.number_of_edges(),
                            'Metric': metric_name,
                            'Fraction_Removed': fraction,
                            'k': k,
                            'Precision': precision,
                            'Recall': recall,
                            'N_Removed': len(removed_edges),
                            'N_Predicted': len(scores)
                        })
                        
                        self.logger.log(
                            f"        k={k}: Precision={precision:.4f}, "
                            f"Recall={recall:.4f}",
                            "DATA"
                        )
                        
                        if current_eval % 10 == 0:
                            self.logger.progress(current_eval, total_evals, "evaluations")
        
        self.logger.end_timer("evaluation")
        
        # Save results
        df_results = pd.DataFrame(results)
        self.results['link_prediction'] = df_results
        self.save_csv(df_results, 'Q4_link_prediction_full.csv')
        
        self.logger.log(
            f"Completed {len(results)} evaluations",
            "SUCCESS"
        )
    
    def create_visualizations(self):
        """Create comprehensive visualizations of results."""
        self.logger.subsection("Creating Visualizations")
        self.logger.start_timer("visualization")
        
        df = self.results['link_prediction']
        
        # Visualization 1: Precision@k by metric
        self.plot_precision_by_metric(df)
        
        # Visualization 2: Precision vs Recall
        self.plot_precision_recall(df)
        
        # Visualization 3: Effect of fraction removed
        self.plot_fraction_effect(df)
        
        self.logger.end_timer("visualization")
    
    def plot_precision_by_metric(self, df):
        """Plot precision comparison for different metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, k in enumerate(self.k_values):
            ax = axes[idx]
            
            df_k = df[df['k'] == k]
            
            # Group by metric and fraction
            df_grouped = df_k.groupby(['Metric', 'Fraction_Removed'])['Precision'].mean().reset_index()
            
            for metric in df_grouped['Metric'].unique():
                df_metric = df_grouped[df_grouped['Metric'] == metric]
                ax.plot(df_metric['Fraction_Removed'], df_metric['Precision'],
                       marker='o', linewidth=2, markersize=8, label=metric)
            
            ax.set_xlabel('Fraction of Edges Removed')
            ax.set_ylabel(f'Average Precision@{k}')
            ax.set_title(f'Precision@{k} by Metric')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        self.save_figure('Q4_precision_by_metric.png')
    
    def plot_precision_recall(self, df):
        """Plot precision-recall curves."""
        fig, axes = plt.subplots(len(self.fractions), len(self.metrics), 
                                figsize=(15, 4*len(self.fractions)))
        
        if len(self.fractions) == 1:
            axes = axes.reshape(1, -1)
        
        for f_idx, fraction in enumerate(self.fractions):
            for m_idx, metric in enumerate(self.metrics.keys()):
                ax = axes[f_idx, m_idx]
                
                df_subset = df[(df['Fraction_Removed'] == fraction) & 
                              (df['Metric'] == metric)]
                
                # Group by k
                df_grouped = df_subset.groupby('k')[['Precision', 'Recall']].mean()
                
                ax.plot(df_grouped['Recall'], df_grouped['Precision'],
                       marker='o', linewidth=2, markersize=8, color='steelblue')
                
                # Annotate k values
                for k in df_grouped.index:
                    prec = df_grouped.loc[k, 'Precision']
                    rec = df_grouped.loc[k, 'Recall']
                    ax.annotate(f'k={k}', (rec, prec), 
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8)
                
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'{metric}\n(fraction={fraction})')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, max(0.1, df_grouped['Recall'].max() * 1.1)])
                ax.set_ylim([0, 1])
        
        plt.tight_layout()
        self.save_figure('Q4_precision_recall_curves.png')
    
    def plot_fraction_effect(self, df):
        """Plot effect of fraction removed on performance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Average across all metrics and k values
        df_avg = df.groupby('Fraction_Removed')[['Precision', 'Recall']].mean()
        
        # Precision
        ax1 = axes[0]
        ax1.plot(df_avg.index, df_avg['Precision'], 
                marker='o', linewidth=2, markersize=10, color='steelblue')
        ax1.set_xlabel('Fraction of Edges Removed')
        ax1.set_ylabel('Average Precision')
        ax1.set_title('Effect of Edge Removal on Precision')
        ax1.grid(True, alpha=0.3)
        
        # Recall
        ax2 = axes[1]
        ax2.plot(df_avg.index, df_avg['Recall'],
                marker='o', linewidth=2, markersize=10, color='coral')
        ax2.set_xlabel('Fraction of Edges Removed')
        ax2.set_ylabel('Average Recall')
        ax2.set_title('Effect of Edge Removal on Recall')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure('Q4_fraction_effect.png')
    
    def print_summary(self):
        """Print comprehensive summary."""
        self.logger.subsection("COMPREHENSIVE SUMMARY")
        
        df = self.results['link_prediction']
        
        self.logger.log(f"\n[STATS] Evaluated {len(self.networks)} networks", "INFO")
        self.logger.log(f"[STATS] Total evaluations: {len(df)}", "INFO")
        
        # Best performing metric
        df_avg = df.groupby('Metric')['Precision'].mean().sort_values(ascending=False)
        
        self.logger.log("\n[STATS] AVERAGE PRECISION BY METRIC:", "INFO")
        for metric, prec in df_avg.items():
            self.logger.log(f"  • {metric}: {prec:.4f}", "INFO")
        
        # Best configuration
        best_row = df.loc[df['Precision'].idxmax()]
        self.logger.log("\n[STATS] BEST CONFIGURATION:", "INFO")
        self.logger.log(
            f"  • Network: {best_row['Network']}", "INFO"
        )
        self.logger.log(
            f"  • Metric: {best_row['Metric']}", "INFO"
        )
        self.logger.log(
            f"  • k={best_row['k']}, fraction={best_row['Fraction_Removed']}", 
            "INFO"
        )
        self.logger.log(
            f"  • Precision: {best_row['Precision']:.4f}, "
            f"Recall: {best_row['Recall']:.4f}",
            "INFO"
        )


def main():
    """Main execution function."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    analysis = Question4Analysis()
    analysis.run()


if __name__ == '__main__':
    main()
