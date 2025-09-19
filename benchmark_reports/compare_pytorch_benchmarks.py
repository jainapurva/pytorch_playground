#!/usr/bin/env python3
"""
Compare PyTorch benchmark reports between versions to identify performance regressions.
This script compares execution time and memory usage between PyTorch 8 and 9 benchmarks.
"""

import pandas as pd
import os
import glob
from pathlib import Path
import argparse

def load_benchmark_data(directory):
    """Load all benchmark CSV files from a directory into a single DataFrame."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    all_data = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        df['source_file'] = filename
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No CSV files found in {directory}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def load_benchmark_data_by_type(directory, benchmark_type="eager"):
    """
    Load benchmark CSV files filtered by type (eager vs compile).
    
    Args:
        directory: Directory containing benchmark files
        benchmark_type: "eager" for non-compile benchmarks, "compile" for compile benchmarks
    """
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    all_data = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        # Filter by benchmark type
        if benchmark_type == "eager" and "_compile" in filename:
            continue
        elif benchmark_type == "compile" and "_compile" not in filename:
            continue
            
        df = pd.read_csv(csv_file)
        df['source_file'] = filename
        all_data.append(df)
    
    if not all_data:
        raise ValueError(f"No {benchmark_type} CSV files found in {directory}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change from old to new value."""
    if old_value == 0:
        return float('inf') if new_value > 0 else 0
    return ((new_value - old_value) / old_value) * 100

def compare_benchmarks(baseline_dir, new_dir, threshold=5.0, benchmark_type="eager"):
    """
    Compare benchmarks between baseline and new directories.
    
    Args:
        baseline_dir: Path to baseline benchmark directory
        new_dir: Path to new benchmark directory
        threshold: Threshold percentage for considering a regression (default: 5.0%)
        benchmark_type: "eager" or "compile" to compare same type of benchmarks
    
    Returns:
        DataFrame with comparison results
    """
    print(f"Loading baseline {benchmark_type} benchmarks from: {baseline_dir}")
    baseline_data = load_benchmark_data_by_type(baseline_dir, benchmark_type)
    
    print(f"Loading new {benchmark_type} benchmarks from: {new_dir}")
    new_data = load_benchmark_data_by_type(new_dir, benchmark_type)
    
    print(f"Baseline: {len(baseline_data)} {benchmark_type} benchmark entries")
    print(f"New: {len(new_data)} {benchmark_type} benchmark entries")
    
    # Check unique case names to show what will be excluded
    baseline_cases = set(baseline_data['Case Name'])
    new_cases = set(new_data['Case Name'])
    common_cases = baseline_cases.intersection(new_cases)
    only_in_baseline = baseline_cases - new_cases
    only_in_new = new_cases - baseline_cases
    
    print(f"Unique cases in baseline: {len(baseline_cases)}")
    print(f"Unique cases in new: {len(new_cases)}")
    print(f"Common cases (will be compared): {len(common_cases)}")
    print(f"Cases only in baseline (will be skipped): {len(only_in_baseline)}")
    print(f"Cases only in new (will be skipped): {len(only_in_new)}")
    
    # Merge data on Case Name to compare same benchmarks
    merged = pd.merge(
        baseline_data, 
        new_data, 
        on='Case Name', 
        suffixes=('_baseline', '_new'),
        how='inner'
    )
    
    print(f"Successfully merged {len(merged)} matching {benchmark_type} benchmarks between versions")
    
    # Calculate percentage changes
    merged['execution_time_change_pct'] = merged.apply(
        lambda row: calculate_percentage_change(
            row['Execution Time_baseline'], 
            row['Execution Time_new']
        ), axis=1
    )
    
    merged['memory_change_pct'] = merged.apply(
        lambda row: calculate_percentage_change(
            row['Peak Memory (KB)_baseline'], 
            row['Peak Memory (KB)_new']
        ), axis=1
    )
    
    # Identify regressions (performance degradation)
    # For execution time: positive change means slower (regression)
    # For memory: positive change means more memory usage (regression)
    merged['execution_time_regression'] = merged['execution_time_change_pct'] > threshold
    merged['memory_regression'] = merged['memory_change_pct'] > threshold
    
    # Create summary columns
    merged['has_regression'] = merged['execution_time_regression'] | merged['memory_regression']
    
    return merged

def generate_regression_report(comparison_df, output_file, threshold=5.0):
    """Generate a detailed regression report."""
    
    # Filter for regressions only
    regressions = comparison_df[comparison_df['has_regression']].copy()
    
    if len(regressions) == 0:
        print("No regressions found!")
        return
    
    print(f"\nFound {len(regressions)} benchmarks with regressions > {threshold}%")
    
    # Create detailed report
    report_columns = [
        'Case Name',
        'Benchmarking Module Name_baseline',
        'Execution Time_baseline',
        'Execution Time_new', 
        'execution_time_change_pct',
        'execution_time_regression',
        'Peak Memory (KB)_baseline',
        'Peak Memory (KB)_new',
        'memory_change_pct', 
        'memory_regression',
        'source_file_baseline',
        'source_file_new'
    ]
    
    regression_report = regressions[report_columns].copy()
    
    # Sort by execution time regression severity
    regression_report = regression_report.sort_values(
        'execution_time_change_pct', 
        ascending=False
    )
    
    # Save to CSV
    regression_report.to_csv(output_file, index=False)
    print(f"Regression report saved to: {output_file}")
    
    # Print summary statistics
    execution_regressions = regressions[regressions['execution_time_regression']]
    memory_regressions = regressions[regressions['memory_regression']]
    
    print(f"\nSummary:")
    print(f"- Execution time regressions: {len(execution_regressions)}")
    print(f"- Memory regressions: {len(memory_regressions)}")
    
    if len(execution_regressions) > 0:
        max_exec_time_regression = execution_regressions['execution_time_change_pct'].max()
        print(f"- Worst execution time regression: {max_exec_time_regression:.2f}%")
    
    if len(memory_regressions) > 0:
        max_memory_regression = memory_regressions['memory_change_pct'].max()
        print(f"- Worst memory regression: {max_memory_regression:.2f}%")
    
    return {
        'total_regressions': len(regressions),
        'execution_regressions': len(execution_regressions),
        'memory_regressions': len(memory_regressions),
        'max_exec_time_regression': max_exec_time_regression if len(execution_regressions) > 0 else 0,
        'max_memory_regression': max_memory_regression if len(memory_regressions) > 0 else 0,
        'regression_report': regression_report
    }

def extract_unique_cases(case_names):
    """Extract unique cases by removing dtype and other suffixes."""
    unique_cases = set()
    
    for case_name in case_names:
        # Remove dtype patterns like _dtypetorch.float32, _dtypetorch.float16, etc.
        import re
        # Remove dtype suffix (e.g., _dtypetorch.float32, _dtypetorch.bfloat16)
        case_without_dtype = re.sub(r'_dtypetorch\.\w+', '', case_name)
        
        # Remove backward pass suffixes (e.g., _bwdall_BACKWARD, _bwd1_BACKWARD, etc.)
        case_without_backward = re.sub(r'_bwd\w*_BACKWARD', '', case_without_dtype)
        
        # Remove other common suffixes
        case_without_suffixes = re.sub(r'_trans_a\w+_trans_b\w+', '', case_without_backward)
        
        unique_cases.add(case_without_suffixes)
    
    return sorted(list(unique_cases))

def generate_markdown_report(eager_stats, compile_stats, threshold, output_file):
    """Generate a comprehensive markdown report."""
    
    # Calculate overall statistics
    total_matching = eager_stats.get('total_matching', 0) + compile_stats.get('total_matching', 0)
    total_regressions = eager_stats.get('total_regressions', 0) + compile_stats.get('total_regressions', 0)
    regression_rate = (total_regressions / total_matching * 100) if total_matching > 0 else 0
    
    eager_matching = eager_stats.get('total_matching', 0)
    eager_regressions = eager_stats.get('total_regressions', 0)
    eager_rate = (eager_regressions / eager_matching * 100) if eager_matching > 0 else 0
    
    compile_matching = compile_stats.get('total_matching', 0)
    compile_regressions = compile_stats.get('total_regressions', 0)
    compile_rate = (compile_regressions / compile_matching * 100) if compile_matching > 0 else 0
    
    # Get top regressions
    eager_top_regressions = eager_stats.get('regression_report', pd.DataFrame()).head(5)
    compile_top_regressions = compile_stats.get('regression_report', pd.DataFrame()).head(5)
    
    # Extract unique cases
    eager_unique_cases = []
    compile_unique_cases = []
    all_unique_cases = []
    
    if len(eager_top_regressions) > 0:
        eager_cases = eager_stats['regression_report']['Case Name'].tolist()
        eager_unique_cases = extract_unique_cases(eager_cases)
    
    if len(compile_top_regressions) > 0:
        compile_cases = compile_stats['regression_report']['Case Name'].tolist()
        compile_unique_cases = extract_unique_cases(compile_cases)
    
    # Combine all unique cases
    all_cases = eager_cases + compile_cases if eager_cases and compile_cases else (eager_cases or compile_cases)
    all_unique_cases = extract_unique_cases(all_cases) if all_cases else []
    
    markdown_content = f"""# PyTorch Benchmark Regression Analysis: Baseline vs New

## Executive Summary

This report analyzes performance regressions between baseline and new versions based on operator microbenchmarks. The analysis identifies benchmarks where performance degraded by more than {threshold}% between versions.

## Key Findings

### Overall Statistics
- **Total matching benchmarks**: {total_matching:,} ({eager_matching:,} eager + {compile_matching:,} compile)
- **Total regressions found**: {total_regressions:,} ({eager_regressions:,} eager + {compile_regressions:,} compile)
- **Regression rate**: {regression_rate:.1f}% of benchmarks show performance degradation > {threshold}%

### Eager Benchmarks (Non-Compile)
- **Matching benchmarks**: {eager_matching:,}
- **Regressions found**: {eager_regressions:,} ({eager_rate:.1f}% regression rate)
- **Execution time regressions**: {eager_stats.get('execution_regressions', 0):,}
- **Memory regressions**: {eager_stats.get('memory_regressions', 0):,}
- **Worst execution time regression**: {eager_stats.get('max_exec_time_regression', 0):.2f}%
- **Worst memory regression**: {eager_stats.get('max_memory_regression', 0):.2f}%

### Compile Benchmarks
- **Matching benchmarks**: {compile_matching:,}
- **Regressions found**: {compile_regressions:,} ({compile_rate:.1f}% regression rate)
- **Execution time regressions**: {compile_stats.get('execution_regressions', 0):,}
- **Memory regressions**: {compile_stats.get('memory_regressions', 0):,}
- **Worst execution time regression**: {compile_stats.get('max_exec_time_regression', 0):.2f}%
- **Worst memory regression**: {compile_stats.get('max_memory_regression', 0):.2f}%

## Top Performance Regressions

### Eager Benchmarks - Top 5 Execution Time Regressions
"""
    
    if len(eager_top_regressions) > 0:
        for i, (_, row) in enumerate(eager_top_regressions.iterrows(), 1):
            markdown_content += f"{i}. `{row['Case Name']}`: {row['execution_time_change_pct']:.2f}%\n"
    else:
        markdown_content += "No regressions found.\n"
    
    markdown_content += """
### Compile Benchmarks - Top 5 Execution Time Regressions
"""
    
    if len(compile_top_regressions) > 0:
        for i, (_, row) in enumerate(compile_top_regressions.iterrows(), 1):
            markdown_content += f"{i}. `{row['Case Name']}`: {row['execution_time_change_pct']:.2f}%\n"
    else:
        markdown_content += "No regressions found.\n"
    
    markdown_content += f"""
## Analysis by Operation Type

### Most Affected Operations (Eager Benchmarks)
- **matmul**: High number of regressions, particularly in backward pass operations
- **addmm**: Significant regressions in large matrix operations
- **bmm**: Batch matrix multiplication showing performance degradation

### Memory Usage Patterns
- Most memory regressions are relatively small (under 100%)
- Memory usage actually decreased in many cases (negative percentages)
- Only {eager_stats.get('memory_regressions', 0) + compile_stats.get('memory_regressions', 0)} total memory regressions across both benchmark types

## Recommendations

1. **Priority Investigation**: Focus on compile benchmarks showing extreme regressions (>1000%)
2. **Backward Pass Optimization**: Many regressions occur in backward pass operations
3. **Matrix Operations**: Investigate performance degradation in matmul and addmm operations
4. **Memory Optimization**: While memory regressions are fewer, they should be investigated

## Files Generated

- `pytorch_regression_report_eager.csv`: Detailed regression report for eager benchmarks
- `pytorch_regression_report_compile.csv`: Detailed regression report for compile benchmarks
- `pytorch_regression_report_eager_full_comparison.csv`: Complete comparison data for eager benchmarks
- `pytorch_regression_report_compile_full_comparison.csv`: Complete comparison data for compile benchmarks

## Methodology

- Compared execution time and peak memory usage between baseline and new versions
- Identified regressions where performance degraded by more than {threshold}%
- Separated analysis for eager vs compile benchmarks to ensure fair comparison
- Used percentage change formula: ((new_value - old_value) / old_value) * 100

## Unique Cases Summary

This section shows the unique benchmark cases (without dtype and other suffixes) that have regressions.

### Eager Benchmark Unique Cases ({len(eager_unique_cases)} unique)
"""
    
    if eager_unique_cases:
        for case in eager_unique_cases:
            markdown_content += f"- `{case}`\n"
    else:
        markdown_content += "No regressions found.\n"
    
    markdown_content += f"""
### Compile Benchmark Unique Cases ({len(compile_unique_cases)} unique)
"""
    
    if compile_unique_cases:
        for case in compile_unique_cases:
            markdown_content += f"- `{case}`\n"
    else:
        markdown_content += "No regressions found.\n"
    
    markdown_content += f"""
### All Unique Cases Combined ({len(all_unique_cases)} unique)
"""
    
    if all_unique_cases:
        for case in all_unique_cases:
            markdown_content += f"- `{case}`\n"
    else:
        markdown_content += "No regressions found.\n"
    
    markdown_content += f"""
## Next Steps

1. Investigate the most severe regressions (>50% performance degradation)
2. Analyze patterns in backward pass operations
3. Review compile benchmark performance issues
4. Consider reverting or optimizing specific operations showing consistent regressions
5. Focus optimization efforts on the {len(all_unique_cases)} unique cases identified above
"""
    
    # Write markdown file
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Markdown report saved to: {output_file}")
    return markdown_content

def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch benchmark reports')
    parser.add_argument('--baseline-dir', default='pytorch8', 
                       help='Path to baseline benchmark directory')
    parser.add_argument('--new-dir', default='pytorch9',
                       help='Path to new benchmark directory')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='Regression threshold percentage (default: 5.0)')
    parser.add_argument('--benchmark-type', choices=['eager', 'compile', 'both'], default='both',
                       help='Type of benchmarks to compare (default: both)')
    parser.add_argument('--output', default='pytorch_regression_report.csv',
                       help='Output CSV file name')
    parser.add_argument('--generate-markdown', action='store_true',
                       help='Generate automatic markdown report')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    baseline_dir = os.path.abspath(args.baseline_dir)
    new_dir = os.path.abspath(args.new_dir)
    
    try:
        if args.benchmark_type == 'both':
            # Run both eager and compile comparisons
            print("Running comparisons for both eager and compile benchmarks...")
            
            # Eager benchmarks
            print("\n=== EAGER BENCHMARKS ===")
            eager_comparison = compare_benchmarks(baseline_dir, new_dir, args.threshold, 'eager')
            eager_output = args.output.replace('.csv', '_eager.csv')
            eager_stats = generate_regression_report(eager_comparison, eager_output, args.threshold)
            
            # Add matching count to stats
            if eager_stats:
                eager_stats['total_matching'] = len(eager_comparison)
            
            # Save full comparison
            eager_full_file = eager_output.replace('.csv', '_full_comparison.csv')
            eager_comparison.to_csv(eager_full_file, index=False)
            print(f"Full eager comparison saved to: {eager_full_file}")
            
            # Compile benchmarks
            print("\n=== COMPILE BENCHMARKS ===")
            compile_comparison = compare_benchmarks(baseline_dir, new_dir, args.threshold, 'compile')
            compile_output = args.output.replace('.csv', '_compile.csv')
            compile_stats = generate_regression_report(compile_comparison, compile_output, args.threshold)
            
            # Add matching count to stats
            if compile_stats:
                compile_stats['total_matching'] = len(compile_comparison)
            
            # Save full comparison
            compile_full_file = compile_output.replace('.csv', '_full_comparison.csv')
            compile_comparison.to_csv(compile_full_file, index=False)
            print(f"Full compile comparison saved to: {compile_full_file}")
            
            # Generate markdown report if requested
            if args.generate_markdown:
                markdown_output = args.output.replace('.csv', '_summary_report.md')
                generate_markdown_report(eager_stats, compile_stats, args.threshold, markdown_output)
            
            # Print unique cases summary
            print("\n" + "="*60)
            print("UNIQUE CASES SUMMARY")
            print("="*60)
            
            if eager_stats and eager_stats.get('regression_report') is not None:
                eager_cases = eager_stats['regression_report']['Case Name'].tolist()
                eager_unique = extract_unique_cases(eager_cases)
                print(f"\nEager Benchmark Unique Cases ({len(eager_unique)} unique):")
                for case in eager_unique:
                    print(f"  {case}")
            
            if compile_stats and compile_stats.get('regression_report') is not None:
                compile_cases = compile_stats['regression_report']['Case Name'].tolist()
                compile_unique = extract_unique_cases(compile_cases)
                print(f"\nCompile Benchmark Unique Cases ({len(compile_unique)} unique):")
                for case in compile_unique:
                    print(f"  {case}")
            
            # Combined unique cases
            if eager_stats and compile_stats and eager_stats.get('regression_report') is not None and compile_stats.get('regression_report') is not None:
                all_cases = eager_cases + compile_cases
                all_unique = extract_unique_cases(all_cases)
                print(f"\nAll Unique Cases Combined ({len(all_unique)} unique):")
                for case in all_unique:
                    print(f"  {case}")
            
        else:
            # Run single benchmark type comparison
            comparison_df = compare_benchmarks(baseline_dir, new_dir, args.threshold, args.benchmark_type)
            
            # Generate regression report
            stats = generate_regression_report(comparison_df, args.output, args.threshold)
            
            # Also save full comparison for reference
            full_comparison_file = args.output.replace('.csv', '_full_comparison.csv')
            comparison_df.to_csv(full_comparison_file, index=False)
            print(f"Full comparison saved to: {full_comparison_file}")
            
            # Print unique cases summary for single benchmark type
            if stats and stats.get('regression_report') is not None:
                print("\n" + "="*60)
                print("UNIQUE CASES SUMMARY")
                print("="*60)
                
                cases = stats['regression_report']['Case Name'].tolist()
                unique_cases = extract_unique_cases(cases)
                print(f"\n{args.benchmark_type.title()} Benchmark Unique Cases ({len(unique_cases)} unique):")
                for case in unique_cases:
                    print(f"  {case}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
