"""
Base classes and utilities for the network analysis homework.
Provides logging, timing, and common analysis structure.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import networkx as nx


class ProgressLogger:
    """Enhanced logging system with timestamps and progress tracking."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.start_time = None
        self.step_times = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        if not self.verbose:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        prefix = {
            "INFO": "[INFO]",
            "SUCCESS": "[SUCCESS]",
            "ERROR": "[ERROR]",
            "WARNING": "[WARNING]",
            "PROGRESS": "[PROGRESS]",
            "DATA": "[DATA]"
        }.get(level, "[LOG]")
        print(f"[{timestamp}] {prefix} {message}")
        sys.stdout.flush()
    
    def section(self, title: str, char="="):
        """Print a section header."""
        if not self.verbose:
            return
        width = 80
        print(f"\n{char * width}")
        print(f"{title.center(width)}")
        print(f"{char * width}\n")
        sys.stdout.flush()
    
    def subsection(self, title: str):
        """Print a subsection header."""
        if not self.verbose:
            return
        print(f"\n{'-' * 80}")
        print(f"  {title}")
        print(f"{'-' * 80}")
        sys.stdout.flush()
    
    def start_timer(self, step_name: str):
        """Start timing a specific step."""
        self.step_times[step_name] = time.time()
        self.log(f"Starting: {step_name}", "PROGRESS")
    
    def end_timer(self, step_name: str):
        """End timing and report duration."""
        if step_name in self.step_times:
            duration = time.time() - self.step_times[step_name]
            self.log(f"Completed: {step_name} (Duration: {duration:.2f}s)", "SUCCESS")
            del self.step_times[step_name]
    
    def progress(self, current: int, total: int, item_name: str = "items"):
        """Log progress for iterative tasks."""
        percent = (current / total) * 100
        self.log(f"Progress: {current}/{total} {item_name} ({percent:.1f}%)", "PROGRESS")


class NetworkAnalysisBase:
    """Base class for all network analysis questions."""
    
    def __init__(self, data_dir: str = "fb100/data", output_dir: str = "results", 
                 verbose: bool = True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.logger = ProgressLogger(verbose)
        self.networks = {}
        self.results = {}
        
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_network(self, filename: str, name: Optional[str] = None) -> nx.Graph:
        """
        Load a single network from GML file.
        
        Args:
            filename: Name of the GML file (e.g., 'Caltech36.gml')
            name: Optional display name (defaults to filename without extension)
        
        Returns:
            NetworkX graph object
        """
        if name is None:
            name = filename.replace('.gml', '')
        
        filepath = os.path.join(self.data_dir, filename)
        
        self.logger.log(f"Loading network: {name} from {filename}")
        
        if not os.path.exists(filepath):
            self.logger.log(f"File not found: {filepath}", "ERROR")
            raise FileNotFoundError(f"Network file not found: {filepath}")
        
        try:
            G = nx.read_gml(filepath)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            
            self.logger.log(
                f"Loaded {name}: {n_nodes:,} nodes, {n_edges:,} edges",
                "SUCCESS"
            )
            
            self.networks[name] = G
            return G
            
        except Exception as e:
            self.logger.log(f"Error loading {filename}: {e}", "ERROR")
            raise
    
    def load_multiple_networks(self, network_files: Dict[str, str]):
        """
        Load multiple networks.
        
        Args:
            network_files: Dict mapping display names to filenames
        """
        self.logger.subsection(f"Loading {len(network_files)} Networks")
        
        for idx, (name, filename) in enumerate(network_files.items(), 1):
            self.logger.progress(idx, len(network_files), "networks")
            self.load_network(filename, name)
    
    def get_largest_component(self, G: nx.Graph) -> nx.Graph:
        """Extract largest connected component from graph."""
        if G.is_directed():
            components = list(nx.weakly_connected_components(G))
        else:
            components = list(nx.connected_components(G))
        
        largest = max(components, key=len)
        lcc = G.subgraph(largest).copy()
        
        self.logger.log(
            f"Largest component: {lcc.number_of_nodes()}/{G.number_of_nodes()} "
            f"nodes ({100*lcc.number_of_nodes()/G.number_of_nodes():.1f}%)",
            "DATA"
        )
        
        return lcc
    
    def save_figure(self, filename: str):
        """Save the current matplotlib figure."""
        import matplotlib.pyplot as plt
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        self.logger.log(f"Saved figure: {filename}", "SUCCESS")
        plt.close()
    
    def save_csv(self, df, filename: str):
        """Save a pandas DataFrame to CSV."""
        import pandas as pd
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        self.logger.log(f"Saved CSV: {filename} ({len(df)} rows)", "SUCCESS")
    
    def run(self):
        """Main execution method - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
