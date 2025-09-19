# PyTorch Benchmark Comparison Tool

This tool compares PyTorch benchmark reports between different versions to identify performance regressions. The initial csv reports are generated using the pytorch/benchmarks/benchmark_operators.

## Quick Start

### Basic Usage

```bash
# Compare both eager and compile benchmarks with automatic markdown report
python compare_pytorch_benchmarks.py --baseline-dir pytorch8 --new-dir pytorch9 --threshold 5.0 --benchmark-type both --generate-markdown --output regression_report.csv
```

### Command Line Options

| Option | Description | Default | Choices |
|--------|-------------|---------|---------|
| `--baseline-dir` | Path to baseline benchmark directory | `pytorch8` | Any directory path |
| `--new-dir` | Path to new benchmark directory | `pytorch9` | Any directory path |
| `--threshold` | Regression threshold percentage | `5.0` | Any positive number |
| `--benchmark-type` | Type of benchmarks to compare | `both` | `eager`, `compile`, `both` |
| `--output` | Output CSV file name | `pytorch_regression_report.csv` | Any filename |
| `--generate-markdown` | Generate automatic markdown report | `False` | Flag (no value) |

## Usage Examples

### 1. Compare Both Benchmark Types (Recommended)

```bash
python compare_pytorch_benchmarks.py \
  --baseline-dir pytorch8 \
  --new-dir pytorch9 \
  --threshold 5.0 \
  --benchmark-type both \
  --generate-markdown \
  --output final_comparison.csv
```

**Output Files:**
- `final_comparison_eager.csv` - Eager benchmark regressions
- `final_comparison_compile.csv` - Compile benchmark regressions
- `final_comparison_eager_full_comparison.csv` - Complete eager data
- `final_comparison_compile_full_comparison.csv` - Complete compile data
- `final_comparison_summary_report.md` - Comprehensive markdown report

### 2. Compare Only Eager Benchmarks

```bash
python compare_pytorch_benchmarks.py \
  --baseline-dir pytorch8 \
  --new-dir pytorch9 \
  --benchmark-type eager \
  --output eager_only.csv
```

### 3. Compare Only Compile Benchmarks

```bash
python compare_pytorch_benchmarks.py \
  --baseline-dir pytorch8 \
  --new-dir pytorch9 \
  --benchmark-type compile \
  --output compile_only.csv
```

### 4. Custom Regression Threshold

```bash
# Use 2% threshold instead of 5%
python compare_pytorch_benchmarks.py \
  --baseline-dir pytorch8 \
  --new-dir pytorch9 \
  --threshold 2.0 \
  --benchmark-type both \
  --generate-markdown \
  --output sensitive_analysis.csv
```

### 5. Different Directory Names

```bash
python compare_pytorch_benchmarks.py \
  --baseline-dir old_version \
  --new-dir new_version \
  --benchmark-type both \
  --generate-markdown \
  --output version_comparison.csv
```

## Understanding the Output

### CSV Files

1. **Regression Reports** (`*_eager.csv`, `*_compile.csv`)
   - Contains only benchmarks with regressions > threshold
   - Shows percentage changes and regression flags
   - Sorted by severity (worst regressions first)

2. **Full Comparison** (`*_full_comparison.csv`)
   - Contains all benchmarks that exist in both versions
   - Includes regressions, improvements, and unchanged benchmarks
   - Complete data for further analysis

### Markdown Report

The auto-generated markdown report includes:
- Executive summary with key statistics
- Detailed breakdown by benchmark type
- Top 5 performance regressions for each type
- **Unique Cases Summary** (simplified case names)
- Analysis and recommendations
- Methodology and next steps

### Console Output

The tool provides real-time feedback:
- Loading progress for each benchmark type
- Validation of common vs unique cases
- Summary statistics
- **Unique Cases Summary** printed to console

## File Structure Requirements

### Input Directory Structure

```
baseline_dir/
├── operator_microbenchmark_add.csv
├── operator_microbenchmark_add_compile.csv
├── operator_microbenchmark_matmul.csv
├── operator_microbenchmark_matmul_compile.csv
├── operator_microbenchmark_bmm.csv
├── operator_microbenchmark_bmm_compile.csv
├── operator_microbenchmark_mm.csv
└── operator_microbenchmark_mm_compile.csv

new_dir/
├── operator_microbenchmark_add.csv
├── operator_microbenchmark_add_compile.csv
├── operator_microbenchmark_matmul.csv
├── operator_microbenchmark_matmul_compile.csv
├── operator_microbenchmark_bmm.csv
├── operator_microbenchmark_bmm_compile.csv
├── operator_microbenchmark_mm.csv
└── operator_microbenchmark_mm_compile.csv
```

### CSV File Format

Each CSV file should contain columns:
- `Benchmarking Framework`
- `Benchmarking Module Name`
- `Case Name`
- `tag`
- `run_backward`
- `Execution Time`
- `Peak Memory (KB)`

## Troubleshooting

### Common Issues

1. **No CSV files found**
   - Ensure directory paths are correct
   - Check that CSV files exist in the specified directories

2. **No matching benchmarks**
   - Verify that both directories contain the same benchmark files
   - Check that `Case Name` columns match between versions

3. **Memory errors with large datasets**
   - Use `--benchmark-type eager` or `--benchmark-type compile` separately
   - Process smaller subsets of data

### Validation

The tool automatically validates:
- Only compares benchmarks that exist in both versions
- Skips benchmarks unique to one version
- Reports how many cases were skipped
- Ensures fair comparison between versions

## Best Practices

1. **Always use `--generate-markdown`** for comprehensive reports
2. **Use `--benchmark-type both`** for complete analysis
3. **Start with 5% threshold** for meaningful regressions
4. **Review unique cases summary** to identify patterns
5. **Check both regression and full comparison files** for complete picture

## Example Workflow

```bash
# 1. Run complete comparison
python compare_pytorch_benchmarks.py \
  --baseline-dir pytorch8 \
  --new-dir pytorch9 \
  --threshold 5.0 \
  --benchmark-type both \
  --generate-markdown \
  --output pytorch_regression_report.csv

# 2. Review the markdown report
cat pytorch_regression_report_summary_report.md

# 3. Examine specific regressions
head -20 pytorch_regression_report_eager.csv
head -20 pytorch_regression_report_compile.csv

# 4. Check unique cases for patterns
grep "Unique Cases" pytorch_regression_report_summary_report.md -A 20
```
