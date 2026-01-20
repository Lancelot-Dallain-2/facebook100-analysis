"""
Utility functions for network analysis.
Includes plotting, statistics, and data processing helpers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import networkx as nx
from scipy import stats


# Set plotting style
sns.set_style("whitegrid")
sns.set_context("notebook")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def list_available_networks(data_dir: str = "fb100/data") -> List[str]:
    """List all available GML files in the data directory."""
    if not os.path.exists(data_dir):
        return []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.gml')]
    return sorted(files)


def get_fb100_metadata() -> pd.DataFrame:
    """
    Get metadata for FB100 networks.
    Returns DataFrame with network name, filename, and estimated size.
    """
    # Common FB100 networks with approximate sizes
    networks = {
        'Caltech': ('Caltech36.gml', 762),
        'MIT': ('MIT8.gml', 6402),
        'Johns Hopkins': ('Johns Hopkins55.gml', 5157),
        'Harvard': ('Harvard1.gml', 15126),
        'Stanford': ('Stanford3.gml', 11586),
        'Princeton': ('Princeton12.gml', 6596),
        'Yale': ('Yale4.gml', 2843),
        'Cornell': ('Cornell5.gml', 5961),
        'Columbia': ('Columbia2.gml', 8512),
        'Dartmouth': ('Dartmouth6.gml', 3824),
    }
    
    data = []
    for name, (filename, approx_size) in networks.items():
        data.append({
            'name': name,
            'filename': filename,
            'approx_nodes': approx_size
        })
    
    return pd.DataFrame(data)


def compute_basic_stats(G: nx.Graph) -> Dict:
    """
    Compute basic statistics for a network.
    
    Returns dictionary with:
        - n_nodes, n_edges
        - density, avg_degree
        - global_clustering, avg_local_clustering
        - diameter (if not too large)
    """
    stats_dict = {}
    
    stats_dict['n_nodes'] = G.number_of_nodes()
    stats_dict['n_edges'] = G.number_of_edges()
    stats_dict['density'] = nx.density(G)
    
    degrees = [d for n, d in G.degree()]
    stats_dict['avg_degree'] = np.mean(degrees)
    stats_dict['median_degree'] = np.median(degrees)
    stats_dict['max_degree'] = np.max(degrees)
    stats_dict['min_degree'] = np.min(degrees)
    
    stats_dict['global_clustering'] = nx.transitivity(G)
    stats_dict['avg_local_clustering'] = nx.average_clustering(G)
    
    # Only compute diameter for smaller networks
    if stats_dict['n_nodes'] < 5000:
        try:
            if nx.is_connected(G):
                stats_dict['diameter'] = nx.diameter(G)
            else:
                lcc = max(nx.connected_components(G), key=len)
                stats_dict['diameter'] = nx.diameter(G.subgraph(lcc))
        except:
            stats_dict['diameter'] = None
    else:
        stats_dict['diameter'] = None
    
    return stats_dict


def plot_degree_distribution(G: nx.Graph, title: str = "Degree Distribution",
                             log_scale: bool = False, ax=None) -> None:
    """
    Plot degree distribution as histogram.
    
    Args:
        G: NetworkX graph
        title: Plot title
        log_scale: Use log-log scale
        ax: Matplotlib axis (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    degrees = [d for n, d in G.degree()]
    
    if log_scale:
        # Log-log binning for power-law distributions
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        ax.scatter(unique_degrees, counts / len(degrees), 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('P(k)')
    else:
        # Regular histogram
        ax.hist(degrees, bins=min(50, max(degrees) - min(degrees) + 1),
               edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_ylabel('Frequency')
    
    ax.set_xlabel('Degree (k)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics as text
    mean_deg = np.mean(degrees)
    median_deg = np.median(degrees)
    ax.text(0.98, 0.98, f'Mean: {mean_deg:.2f}\nMedian: {median_deg:.2f}',
           transform=ax.transAxes, verticalalignment='top',
           horizontalalignment='right', bbox=dict(boxstyle='round', 
           facecolor='white', alpha=0.8))


def plot_scatter(x: List, y: List, title: str, xlabel: str, ylabel: str,
                log_scale: bool = False, show_correlation: bool = True,
                ax=None) -> None:
    """
    Create scatter plot with optional correlation coefficient.
    
    Args:
        x, y: Data arrays
        title: Plot title
        xlabel, ylabel: Axis labels
        log_scale: Use log scale for both axes
        show_correlation: Display Pearson correlation
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.scatter(x, y, alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add correlation if requested
    if show_correlation and len(x) > 1:
        # Remove any NaN or inf values
        valid_idx = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid_idx) > 1:
            corr, p_value = stats.pearsonr(np.array(x)[valid_idx], 
                                          np.array(y)[valid_idx])
            ax.text(0.05, 0.95, f'r = {corr:.4f}\np = {p_value:.4e}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def compute_assortativity(G: nx.Graph, attribute: str) -> Optional[float]:
    """
    Compute assortativity coefficient for a node attribute.
    
    Args:
        G: NetworkX graph
        attribute: Node attribute name
    
    Returns:
        Assortativity coefficient or None if cannot be computed
    """
    try:
        # Check if attribute exists for nodes
        has_attr = sum(1 for n in G.nodes() if attribute in G.nodes[n]) > 0
        if not has_attr:
            return None
        
        # For degree assortativity
        if attribute == 'degree':
            return nx.degree_assortativity_coefficient(G)
        
        # For categorical attributes
        return nx.attribute_assortativity_coefficient(G, attribute)
    
    except:
        return None


def analyze_node_attributes(G: nx.Graph) -> Dict:
    """
    Analyze what attributes are available in the network nodes.
    
    Returns dictionary with:
        - attribute names
        - coverage (% of nodes with attribute)
        - unique values count
        - sample values
    """
    if G.number_of_nodes() == 0:
        return {}
    
    # Get all possible attributes from first node
    sample_node = list(G.nodes())[0]
    attributes = list(G.nodes[sample_node].keys()) if G.nodes[sample_node] else []
    
    attr_analysis = {}
    
    for attr in attributes:
        values = []
        count = 0
        
        for node in G.nodes():
            if attr in G.nodes[node]:
                count += 1
                values.append(G.nodes[node][attr])
        
        coverage = count / G.number_of_nodes()
        unique_values = len(set(values))
        
        attr_analysis[attr] = {
            'coverage': coverage,
            'unique_values': unique_values,
            'sample_values': list(set(values))[:10]  # First 10 unique values
        }
    
    return attr_analysis


def filter_incomplete_attributes(G: nx.Graph, min_coverage: float = 0.5) -> List[str]:
    """
    Get list of attributes with sufficient coverage.
    
    Args:
        G: NetworkX graph
        min_coverage: Minimum fraction of nodes that must have the attribute
    
    Returns:
        List of attribute names with sufficient coverage
    """
    attr_analysis = analyze_node_attributes(G)
    return [attr for attr, info in attr_analysis.items() 
            if info['coverage'] >= min_coverage]


def safe_divide(numerator, denominator, default=0.0):
    """Safely divide two numbers, returning default if denominator is 0."""
    if denominator == 0:
        return default
    return numerator / denominator


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
