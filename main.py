"""
Main Execution Script for Network Analysis Homework
NET 4103/7431

This script provides a unified interface to run all homework questions
individually or all together.

Usage:
    python main.py --all              # Run all questions
    python main.py --q2               # Run only Question 2
    python main.py --q2 --q3          # Run Questions 2 and 3
    python main.py --q3 --q4 --q5 --q6  # Run Questions 3-6
"""

import os
import sys
import argparse
import time
from datetime import datetime
import subprocess


class HomeworkRunner:
    """Manager for running homework analysis scripts."""
    
    def __init__(self):
        self.scripts = {
            'q2': 'q2_analysis.py',
            'q3': 'q3_assortativity.py',
            'q4': 'q4_link_prediction.py',
            'q5': 'q5_label_propagation.py',
            'q6': 'q6_community.py'
        }
        
        self.descriptions = {
            'q2': 'Social Network Analysis (degree distribution, clustering, density)',
            'q3': 'Assortativity Analysis (5 attributes across all networks) - LONG RUNTIME',
            'q4': 'Link Prediction (Common Neighbors, Jaccard, Adamic/Adar)',
            'q5': 'Label Propagation (missing attribute recovery)',
            'q6': 'Community Detection (research question validation)'
        }
        
        self.estimated_times = {
            'q2': '2-5 minutes',
            'q3': '20-60 minutes',
            'q4': '15-30 minutes',
            'q5': '10-25 minutes',
            'q6': '5-15 minutes'
        }
    
    def print_header(self):
        """Print program header."""
        print("\n" + "="*80)
        print(" "*20 + "NET 4103/7431 HOMEWORK - NETWORK ANALYSIS")
        print(" "*25 + "Facebook100 Dataset Analysis")
        print("="*80 + "\n")
    
    def print_question_info(self, questions: list):
        """Print information about questions to be run."""
        print("\n" + "-"*80)
        print("EXECUTION PLAN")
        print("-"*80)
        
        total_time_min = 0
        total_time_max = 0
        
        for q in questions:
            print(f"\n{q.upper()}: {self.descriptions[q]}")
            print(f"  Script: {self.scripts[q]}")
            print(f"  Estimated time: {self.estimated_times[q]}")
            
            # Parse time range for total calculation
            time_range = self.estimated_times[q].replace(' minutes', '').split('-')
            total_time_min += int(time_range[0])
            if len(time_range) > 1:
                total_time_max += int(time_range[1])
            else:
                total_time_max += int(time_range[0])
        
        print(f"\nTOTAL ESTIMATED TIME: {total_time_min}-{total_time_max} minutes")
        print("-"*80 + "\n")
    
    def run_script(self, question: str) -> bool:
        """
        Run a single question script.
        
        Args:
            question: Question identifier (e.g., 'q2')
        
        Returns:
            True if successful, False otherwise
        """
        script = self.scripts[question]
        
        if not os.path.exists(script):
            print(f"[ERROR] Script not found: {script}")
            return False
        
        print("\n" + "="*80)
        print(f"RUNNING {question.upper()}: {self.descriptions[question]}")
        print("="*80)
        print(f"Script: {script}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-"*80 + "\n")
        
        start_time = time.time()
        
        try:
            # Run script with real-time output
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')
                sys.stdout.flush()
            
            process.wait()
            
            duration = time.time() - start_time
            
            if process.returncode == 0:
                print("\n" + "-"*80)
                print(f"[SUCCESS] {question.upper()} completed successfully")
                print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
                print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80 + "\n")
                return True
            else:
                print("\n" + "-"*80)
                print(f"[ERROR] {question.upper()} failed with exit code {process.returncode}")
                print(f"Duration: {duration:.2f} seconds")
                print("="*80 + "\n")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n[ERROR] Exception while running {question}: {e}")
            print(f"Duration before error: {duration:.2f} seconds")
            return False
    
    def run_all(self, questions: list):
        """
        Run multiple questions in sequence.
        
        Args:
            questions: List of question identifiers
        """
        self.print_header()
        self.print_question_info(questions)
        
        # Confirm execution
        response = input("Proceed with execution? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("\nExecution cancelled.")
            return
        
        print("\n" + "="*80)
        print("STARTING EXECUTION")
        print("="*80)
        print(f"Overall start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        overall_start = time.time()
        results = {}
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Processing {question.upper()}...")
            success = self.run_script(question)
            results[question] = success
            
            # Brief pause between scripts
            if i < len(questions):
                time.sleep(2)
        
        overall_duration = time.time() - overall_start
        
        # Print final summary
        self.print_final_summary(results, overall_duration)
    
    def print_final_summary(self, results: dict, duration: float):
        """Print final execution summary."""
        print("\n" + "="*80)
        print(" "*30 + "EXECUTION SUMMARY")
        print("="*80 + "\n")
        
        successful = [q for q, success in results.items() if success]
        failed = [q for q, success in results.items() if not success]
        
        print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nQuestions run: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            print("\n[OK] SUCCESSFUL:")
            for q in successful:
                print(f"  • {q.upper()}: {self.descriptions[q]}")
        
        if failed:
            print("\n[FAIL] FAILED:")
            for q in failed:
                print(f"  • {q.upper()}: {self.descriptions[q]}")
        
        print("\n" + "="*80)
        print("\nResults saved in respective results/ subdirectories:")
        print("  • results/q2/ - Question 2 outputs")
        print("  • results/q3/ - Question 3 outputs")
        print("  • results/q4/ - Question 4 outputs")
        print("  • results/q5/ - Question 5 outputs")
        print("  • results/q6/ - Question 6 outputs")
        
        print("\nCheck individual CSV files and PNG visualizations for detailed results.")
        print("="*80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run NET 4103/7431 Homework Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all              # Run all questions
  python main.py --q2               # Run only Question 2
  python main.py --q2 --q3          # Run Questions 2 and 3
  python main.py --q3 --q4 --q5 --q6  # Run Questions 3-6
  
Notes:
  • Question 3 (Assortativity) has the longest runtime (~20-60 min)
  • All scripts provide detailed progress logging
  • Results are saved automatically to results/ directories
  • Visualizations are saved as PNG files
  • Data tables are saved as CSV files
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all questions (Q2-Q6)')
    parser.add_argument('--q2', action='store_true',
                       help='Run Question 2 (Social Network Analysis)')
    parser.add_argument('--q3', action='store_true',
                       help='Run Question 3 (Assortativity) - LONG RUNTIME')
    parser.add_argument('--q4', action='store_true',
                       help='Run Question 4 (Link Prediction)')
    parser.add_argument('--q5', action='store_true',
                       help='Run Question 5 (Label Propagation)')
    parser.add_argument('--q6', action='store_true',
                       help='Run Question 6 (Community Detection)')
    
    args = parser.parse_args()
    
    # Determine which questions to run
    runner = HomeworkRunner()
    questions_to_run = []
    
    if args.all:
        questions_to_run = ['q2', 'q3', 'q4', 'q5', 'q6']
    else:
        if args.q2:
            questions_to_run.append('q2')
        if args.q3:
            questions_to_run.append('q3')
        if args.q4:
            questions_to_run.append('q4')
        if args.q5:
            questions_to_run.append('q5')
        if args.q6:
            questions_to_run.append('q6')
    
    if not questions_to_run:
        print("No questions selected. Use --all or specify individual questions.")
        print("Run with --help for usage information.")
        sys.exit(1)
    
    # Run selected questions
    runner.run_all(questions_to_run)


if __name__ == '__main__':
    main()

