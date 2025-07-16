"""
Script to run benchmarks comparing the original and optimized versions of the BPM Analysis application.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Optional
import json

# Import benchmark utilities
from benchmark_utils import (
    BenchmarkRunner, 
    PerformanceMetrics, 
    compare_versions,
    create_test_suite
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)

def find_audio_files(directory: str) -> List[str]:
    """Find all audio files in a directory."""
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def run_original_benchmarks(test_files: List[str], params: Dict, output_dir: str) -> PerformanceMetrics:
    """Run benchmarks on the original version."""
    logging.info("Running benchmarks on original version...")
    
    # Import the original version and default parameters
    from bpm_analysis import analyze_wav_file
    from config import DEFAULT_PARAMS
    
    # Create benchmark runner
    runner = BenchmarkRunner(test_files, output_dir=output_dir)
    
    # Use default parameters if none provided
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Create a wrapper function to handle the required arguments
    def analyze_wrapper(file_path, params):
        return analyze_wav_file(
            file_path, 
            params=params, 
            start_bpm_hint=None,
            original_file_path=file_path,
            output_directory=output_dir
        )
    
    # Run benchmarks
    metrics = runner.run_benchmarks(analyze_wrapper, params)
    
    # Save results
    runner.save_results("original")
    
    return metrics

def run_optimized_benchmarks(test_files: List[str], params: Dict, output_dir: str) -> PerformanceMetrics:
    """Run benchmarks on the optimized version."""
    logging.info("Running benchmarks on optimized version...")
    
    # Import the optimized version and default parameters
    from bpm_analysis_optimized import analyze_wav_file
    from config import DEFAULT_PARAMS
    
    # Create benchmark runner
    runner = BenchmarkRunner(test_files, output_dir=output_dir)
    
    # Use default parameters if none provided
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    # Create a wrapper function to handle the required arguments
    def analyze_wrapper(file_path, params):
        return analyze_wav_file(
            file_path, 
            params=params, 
            start_bpm_hint=None,
            original_file_path=file_path,
            output_directory=output_dir
        )
    
    # Run benchmarks
    metrics = runner.run_benchmarks(analyze_wrapper, params)
    
    # Save results
    runner.save_results("optimized")
    
    return metrics

def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Run benchmarks for BPM Analysis")
    parser.add_argument("--test-dir", default="benchmark_tests", 
                      help="Directory containing test audio files")
    parser.add_argument("--output-dir", default="benchmark_results",
                      help="Directory to save benchmark results")
    parser.add_argument("--create-test-suite", action="store_true",
                      help="Create a test suite from sample files")
    parser.add_argument("--sample-dir", default="processed_files",
                      help="Directory containing sample audio files")
    parser.add_argument("--params-file", default=None,
                      help="JSON file containing parameters for analysis")
    parser.add_argument("--original-only", action="store_true",
                      help="Only run benchmarks on the original version")
    parser.add_argument("--optimized-only", action="store_true",
                      help="Only run benchmarks on the optimized version")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test suite if requested
    if args.create_test_suite:
        test_files = create_test_suite(args.sample_dir, args.test_dir)
    else:
        # Find test files
        test_files = find_audio_files(args.test_dir)
    
    if not test_files:
        logging.error(f"No audio files found in {args.test_dir}")
        sys.exit(1)
    
    logging.info(f"Found {len(test_files)} audio files for benchmarking")
    
    # Load parameters if provided
    params = None
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                params = json.load(f)
            logging.info(f"Loaded parameters from {args.params_file}")
        except Exception as e:
            logging.error(f"Error loading parameters: {e}")
            sys.exit(1)
    
    # Run benchmarks
    original_metrics = None
    optimized_metrics = None
    
    if not args.optimized_only:
        try:
            original_metrics = run_original_benchmarks(test_files, params, args.output_dir)
        except ImportError:
            logging.error("Could not import original version (bpm_analysis.py)")
            if args.original_only:
                sys.exit(1)
    
    if not args.original_only:
        try:
            optimized_metrics = run_optimized_benchmarks(test_files, params, args.output_dir)
        except ImportError:
            logging.error("Could not import optimized version (bpm_analysis_optimized.py)")
            if args.optimized_only:
                sys.exit(1)
    
    # Compare versions if both were benchmarked
    if original_metrics and optimized_metrics:
        report_path = compare_versions(original_metrics, optimized_metrics, args.output_dir)
        logging.info(f"Comparison report saved to {report_path}")

if __name__ == "__main__":
    main()