# BPM Analysis Benchmarking Infrastructure

This directory contains tools for benchmarking and profiling the BPM Analysis application. The benchmarking infrastructure allows you to measure performance metrics such as execution time, memory usage, and CPU utilization before and after optimizations.

## Overview

The benchmarking infrastructure consists of the following components:

1. **benchmark_utils.py**: A library of utilities for measuring performance metrics and comparing results.
2. **run_benchmarks.py**: A script to run benchmarks on the original and optimized versions of the BPM Analysis application.
3. **benchmark_tests/**: A directory containing test audio files for benchmarking.
4. **benchmark_results/**: A directory containing benchmark results and comparison reports.

## Usage

### Creating a Test Suite

To create a test suite from existing audio files:

```bash
python run_benchmarks.py --create-test-suite --sample-dir processed_files
```

This will copy audio files from the specified directory to the `benchmark_tests` directory.

### Running Benchmarks

To run benchmarks on the original version only:

```bash
python run_benchmarks.py --original-only
```

To run benchmarks on the optimized version only:

```bash
python run_benchmarks.py --optimized-only
```

To run benchmarks on both versions and generate a comparison report:

```bash
python run_benchmarks.py
```

### Specifying Parameters

You can specify custom parameters for the analysis by providing a JSON file:

```bash
python run_benchmarks.py --params-file custom_params.json
```

### Output

The benchmarking script generates the following outputs in the `benchmark_results` directory:

1. **original_metrics.csv**: Performance metrics for the original version.
2. **optimized_metrics.csv**: Performance metrics for the optimized version.
3. **performance_comparison.png**: A bar chart comparing the performance of the original and optimized versions.
4. **comparison_report.md**: A detailed report comparing the performance of the original and optimized versions.

## Benchmark Metrics

The benchmarking infrastructure measures the following metrics:

1. **Execution Time**: The time taken to execute the analysis function, measured in seconds.
2. **Memory Usage**: The peak memory usage during execution, measured in megabytes (MB).
3. **CPU Usage**: The average CPU utilization during execution, measured as a percentage.

## Adding Custom Benchmarks

To add custom benchmarks, you can modify the `run_benchmarks.py` script to include additional functions or components to benchmark. The `BenchmarkRunner` class in `benchmark_utils.py` provides methods for measuring performance metrics for any function.

Example:

```python
# Create a benchmark runner
runner = BenchmarkRunner(test_files, output_dir="benchmark_results")

# Benchmark a specific function
result = runner.benchmark_function(
    "my_custom_function",
    my_custom_function,
    arg1, arg2, kwarg1=value1
)
```

## Profiling Decorators

The `benchmark_utils.py` module includes a `profile_function` decorator that can be used to profile individual functions:

```python
from benchmark_utils import profile_function

@profile_function
def my_function():
    # Function code here
    pass
```

This will log the execution time of the function each time it is called.

## Comparing Versions

The `compare_versions` function in `benchmark_utils.py` can be used to generate a comparison report between two sets of performance metrics:

```python
from benchmark_utils import compare_versions, PerformanceMetrics

# Load or create performance metrics
original_metrics = PerformanceMetrics()
optimized_metrics = PerformanceMetrics()

# Generate comparison report
report_path = compare_versions(original_metrics, optimized_metrics)
```

## Best Practices

1. **Use consistent test data**: Always use the same test files for benchmarking both the original and optimized versions to ensure fair comparisons.
2. **Run multiple iterations**: For more accurate results, run benchmarks multiple times and average the results.
3. **Minimize background processes**: Close unnecessary applications and processes before running benchmarks to reduce interference.
4. **Profile specific components**: Use the `benchmark_function` method to profile specific components or functions to identify bottlenecks.
5. **Document optimization changes**: Keep track of optimization changes and their impact on performance metrics.

## Troubleshooting

If you encounter issues with the benchmarking infrastructure, check the following:

1. **Missing dependencies**: Ensure all required dependencies are installed by running `pip install -r requirements.txt`.
2. **File permissions**: Ensure the script has permission to read test files and write to the output directory.
3. **Memory errors**: If you encounter memory errors, try reducing the number or size of test files.
4. **Import errors**: Ensure both the original and optimized versions of the BPM Analysis application are available in the Python path.

## Contributing

When adding new benchmarks or optimizations, please follow these guidelines:

1. **Document changes**: Add comments explaining the purpose and expected impact of optimizations.
2. **Maintain compatibility**: Ensure optimized code maintains the same API and functionality as the original code.
3. **Add test cases**: Include test cases that verify the correctness of optimized code.
4. **Benchmark before and after**: Always benchmark both before and after optimization to measure the impact.
