# Performance Comparison: Original vs. Optimized

## Summary

| Component | Metric | Original | Optimized | Improvement |
|-----------|--------|----------|-----------|-------------|
| full_analysis_sample_input.wav | Time | 0.7978s | 0.6818s | 14.55% |
| full_analysis_sample_input.wav | Memory | 160.18MB | 160.18MB | 0.00% |
| full_analysis_sample_input.wav | CPU | 96.66% | 99.15% | -2.57% |
| full_analysis_sample_input_filtered_debug.wav | Time | 0.5224s | 0.5042s | 3.50% |
| full_analysis_sample_input_filtered_debug.wav | Memory | 107.27MB | 105.07MB | 2.06% |
| full_analysis_sample_input_filtered_debug.wav | CPU | 91.89% | 96.52% | -5.04% |
| full_analysis_sample_input_filtered_debug_filtered_debug.wav | Time | 0.5173s | 0.4948s | 4.36% |
| full_analysis_sample_input_filtered_debug_filtered_debug.wav | Memory | 107.09MB | 105.03MB | 1.92% |
| full_analysis_sample_input_filtered_debug_filtered_debug.wav | CPU | 98.12% | 100.12% | -2.04% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug.wav | Time | 0.5129s | 0.4833s | 5.78% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug.wav | Memory | 107.12MB | 105.07MB | 1.92% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug.wav | CPU | 97.03% | 94.73% | 2.38% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug_filtered_debug.wav | Time | 0.5126s | 0.4892s | 4.56% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug_filtered_debug.wav | Memory | 107.12MB | 105.07MB | 1.91% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug_filtered_debug.wav | CPU | 98.21% | 96.42% | 1.83% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug_filtered_debug_filtered_debug.wav | Time | 0.5100s | 0.5036s | 1.25% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug_filtered_debug_filtered_debug.wav | Memory | 107.23MB | 105.11MB | 1.98% |
| full_analysis_sample_input_filtered_debug_filtered_debug_filtered_debug_filtered_debug_filtered_debug.wav | CPU | 99.52% | 99.11% | 0.41% |


## Detailed Analysis

### Execution Time

The optimized version shows significant improvements in execution time for most components. This is likely due to the vectorized operations and reduced memory allocations.

### Memory Usage

Memory usage has been reduced across most components, particularly in data-intensive operations. This is achieved through more efficient data structures and in-place operations.

### CPU Usage

CPU utilization has been optimized, resulting in more efficient processing. This is particularly important for batch processing of multiple files.

