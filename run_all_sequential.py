#!/usr/bin/env python3
"""
Script pour exécuter toutes les questions séquentiellement avec vérification des résultats.
"""

import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

class SequentialRunner:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "results"
        self.questions = [
            {"name": "Q2", "script": "q2_analysis.py", "expected_files": [
                "results/q2/Q2a_degree_distributions.png",
                "results/q2/Q2b_clustering_density.csv",
                "results/q2/Q2b_clustering_density_viz.png",
                "results/q2/Q2c_correlations.csv",
                "results/q2/Q2c_degree_vs_clustering.png"
            ]},
            {"name": "Q3", "script": "q3_assortativity.py", "expected_files": [
                "results/q3/Q3_assortativity_results.csv",
                "results/q3/Q3_assortativity_distributions.png"
            ]},
            {"name": "Q4", "script": "q4_link_prediction.py", "expected_files": [
                "results/q4/Q4_link_prediction_results.csv",
                "results/q4/Q4_precision_curves.png",
                "results/q4/Q4_recall_curves.png"
            ]},
            {"name": "Q5", "script": "q5_label_propagation.py", "expected_files": [
                "results/q5/Q5_label_propagation_results.csv",
                "results/q5/Q5_accuracy_curves.png"
            ]},
            {"name": "Q6", "script": "q6_community.py", "expected_files": [
                "results/q6/Q6_community_detection_results.csv",
                "results/q6/Q6_homophily_analysis.csv"
            ]}
        ]
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{level}] {message}")
        
    def check_files_exist(self, file_list):
        """Vérifie que tous les fichiers attendus existent."""
        missing = []
        for filepath in file_list:
            full_path = self.base_dir / filepath
            if not full_path.exists():
                missing.append(filepath)
        return missing
        
    def run_question(self, question):
        """Exécute une question et vérifie les résultats."""
        self.log(f"{'='*80}")
        self.log(f"STARTING {question['name']}: {question['script']}")
        self.log(f"{'='*80}")
        
        script_path = self.base_dir / question['script']
        if not script_path.exists():
            self.log(f"ERROR: Script not found: {script_path}", "ERROR")
            return False
            
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.base_dir),
                capture_output=False,
                text=True,
                check=False
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                self.log(f"ERROR: {question['name']} failed with exit code {result.returncode}", "ERROR")
                return False
                
            self.log(f"SUCCESS: {question['name']} completed in {elapsed:.2f}s", "SUCCESS")
            
            # Vérifier que les fichiers attendus ont été créés
            self.log(f"Verifying output files for {question['name']}...")
            missing = self.check_files_exist(question['expected_files'])
            
            if missing:
                self.log(f"WARNING: Missing expected files: {missing}", "WARN")
            else:
                self.log(f"SUCCESS: All {len(question['expected_files'])} expected files created", "SUCCESS")
                
            return True
            
        except KeyboardInterrupt:
            self.log(f"INTERRUPTED: User cancelled {question['name']}", "WARN")
            raise
        except Exception as e:
            self.log(f"ERROR: Exception during {question['name']}: {e}", "ERROR")
            return False
            
    def run_all(self, start_from="Q2"):
        """Exécute toutes les questions séquentiellement."""
        self.log("="*80)
        self.log("SEQUENTIAL EXECUTION OF ALL QUESTIONS")
        self.log("="*80)
        
        start_idx = 0
        for idx, q in enumerate(self.questions):
            if q['name'] == start_from:
                start_idx = idx
                break
                
        total_start = time.time()
        completed = []
        failed = []
        
        for question in self.questions[start_idx:]:
            success = self.run_question(question)
            
            if success:
                completed.append(question['name'])
            else:
                failed.append(question['name'])
                self.log(f"STOPPING: {question['name']} failed", "ERROR")
                break
                
            self.log("")  # Ligne vide entre questions
            
        total_elapsed = time.time() - total_start
        
        # Résumé final
        self.log("="*80)
        self.log("FINAL SUMMARY")
        self.log("="*80)
        self.log(f"Total execution time: {total_elapsed/60:.2f} minutes")
        self.log(f"Completed: {len(completed)} - {', '.join(completed)}")
        if failed:
            self.log(f"Failed: {len(failed)} - {', '.join(failed)}", "ERROR")
        else:
            self.log("All questions completed successfully!", "SUCCESS")
            
        return len(failed) == 0

def main():
    runner = SequentialRunner()
    
    # Déterminer depuis quelle question commencer
    start_from = "Q2"
    if len(sys.argv) > 1:
        start_from = sys.argv[1].upper()
        if not start_from.startswith("Q"):
            start_from = f"Q{start_from}"
            
    success = runner.run_all(start_from=start_from)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
