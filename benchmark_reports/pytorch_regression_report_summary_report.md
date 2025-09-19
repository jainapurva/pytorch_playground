# PyTorch Benchmark Regression Analysis: Baseline vs New

## Executive Summary

This report analyzes performance regressions between baseline and new versions based on operator microbenchmarks. The analysis identifies benchmarks where performance degraded by more than 5.0% between versions.

## Key Findings

### Overall Statistics
- **Total matching benchmarks**: 2,933 (1,460 eager + 1,473 compile)
- **Total regressions found**: 1,021 (411 eager + 610 compile)
- **Regression rate**: 34.8% of benchmarks show performance degradation > 5.0%

### Eager Benchmarks (Non-Compile)
- **Matching benchmarks**: 1,460
- **Regressions found**: 411 (28.2% regression rate)
- **Execution time regressions**: 405
- **Memory regressions**: 8
- **Worst execution time regression**: 75.13%
- **Worst memory regression**: 77.78%

### Compile Benchmarks
- **Matching benchmarks**: 1,473
- **Regressions found**: 610 (41.4% regression rate)
- **Execution time regressions**: 606
- **Memory regressions**: 8
- **Worst execution time regression**: 79416.98%
- **Worst memory regression**: 77.78%

## Top Performance Regressions

### Eager Benchmarks - Top 5 Execution Time Regressions
1. `addmm_M1024_N512_K512_cuda_dtypetorch.float16_bwdall_BACKWARD`: 75.13%
2. `matmul_M256_N512_K4096_trans_aFalse_trans_bTrue_cuda_dtypetorch.bfloat16_bwd1_BACKWARD`: 67.57%
3. `matmul_M256_N512_K512_trans_aFalse_trans_bFalse_cuda_dtypetorch.bfloat16_bwdall_BACKWARD`: 64.87%
4. `matmul_M256_N512_K4096_trans_aFalse_trans_bTrue_cuda_dtypetorch.float16_bwdall_BACKWARD`: 63.24%
5. `matmul_M256_N512_K4096_trans_aTrue_trans_bTrue_cuda_dtypetorch.float32_bwdall_BACKWARD`: 62.72%

### Compile Benchmarks - Top 5 Execution Time Regressions
1. `add_M128_N32_K512_cuda_dtypetorch.float32`: 79416.98%
2. `add_M8_N64_K512_cuda_dtypetorch.float32`: 77588.86%
3. `bmm_B8_M256_N256_K64_cuda_dtypetorch.float16`: 32551.59%
4. `mm_M256_N512_K512_cuda_dtypetorch.float32`: 23151.49%
5. `matmul_M256_N512_K512_trans_aFalse_trans_bTrue_cuda_dtypetorch.float32`: 22762.15%

## Analysis by Operation Type

### Most Affected Operations (Eager Benchmarks)
- **matmul**: High number of regressions, particularly in backward pass operations
- **addmm**: Significant regressions in large matrix operations
- **bmm**: Batch matrix multiplication showing performance degradation

### Memory Usage Patterns
- Most memory regressions are relatively small (under 100%)
- Memory usage actually decreased in many cases (negative percentages)
- Only 16 total memory regressions across both benchmark types

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
- Identified regressions where performance degraded by more than 5.0%
- Separated analysis for eager vs compile benchmarks to ensure fair comparison
- Used percentage change formula: ((new_value - old_value) / old_value) * 100

## Unique Cases Summary

This section shows the unique benchmark cases (without dtype and other suffixes) that have regressions.

### Eager Benchmark Unique Cases (68 unique)
- `addbmm_B32_M1024_N1024_K128_cuda`
- `addbmm_B32_M1024_N1024_K64_cuda`
- `addbmm_B32_M1024_N256_K128_cuda`
- `addbmm_B32_M1024_N256_K64_cuda`
- `addbmm_B32_M256_N1024_K128_cuda`
- `addbmm_B32_M256_N1024_K64_cuda`
- `addbmm_B32_M256_N256_K128_cuda`
- `addbmm_B32_M256_N256_K64_cuda`
- `addmm_M1024_N4096_K4096_cuda`
- `addmm_M1024_N4096_K512_cuda`
- `addmm_M1024_N512_K4096_cuda`
- `addmm_M1024_N512_K512_cuda`
- `addmm_M256_N4096_K4096_cuda`
- `addmm_M256_N4096_K512_cuda`
- `addmm_M256_N512_K4096_cuda`
- `addmm_M256_N512_K512_cuda`
- `addmm_M3000_N4096_K512_cuda`
- `addmm_M3000_N512_K4096_cuda`
- `addmm_M3000_N512_K512_cuda`
- `baddbmm_B32_M1024_N256_K128_cuda`
- `baddbmm_B32_M1024_N256_K64_cuda`
- `baddbmm_B32_M256_N1024_K128_cuda`
- `baddbmm_B32_M256_N1024_K64_cuda`
- `baddbmm_B32_M256_N256_K128_cuda`
- `baddbmm_B32_M256_N256_K64_cuda`
- `baddbmm_B8_M1024_N1024_K128_cuda`
- `baddbmm_B8_M1024_N1024_K64_cuda`
- `baddbmm_B8_M1024_N256_K128_cuda`
- `baddbmm_B8_M1024_N256_K64_cuda`
- `baddbmm_B8_M256_N1024_K128_cuda`
- `baddbmm_B8_M256_N1024_K64_cuda`
- `baddbmm_B8_M256_N256_K128_cuda`
- `baddbmm_B8_M256_N256_K64_cuda`
- `bmm_B32_M1024_N256_K128_cuda`
- `bmm_B32_M1024_N256_K64_cuda`
- `bmm_B32_M256_N1024_K128_cuda`
- `bmm_B32_M256_N1024_K64_cuda`
- `bmm_B32_M256_N256_K128_cuda`
- `bmm_B32_M256_N256_K64_cuda`
- `bmm_B8_M1024_N1024_K128_cuda`
- `bmm_B8_M1024_N1024_K64_cuda`
- `bmm_B8_M1024_N256_K128_cuda`
- `bmm_B8_M1024_N256_K64_cuda`
- `bmm_B8_M256_N1024_K128_cuda`
- `bmm_B8_M256_N1024_K64_cuda`
- `bmm_B8_M256_N256_K128_cuda`
- `bmm_B8_M256_N256_K64_cuda`
- `matmul_M1024_N4096_K4096`
- `matmul_M1024_N4096_K512`
- `matmul_M1024_N512_K4096`
- `matmul_M1024_N512_K512`
- `matmul_M256_N4096_K4096`
- `matmul_M256_N4096_K512`
- `matmul_M256_N512_K4096`
- `matmul_M256_N512_K512`
- `matmul_M3000_N4096_K512`
- `matmul_M3000_N512_K4096`
- `matmul_M3000_N512_K512`
- `mm_M1024_N4096_K512_cuda`
- `mm_M1024_N512_K4096_cuda`
- `mm_M1024_N512_K512_cuda`
- `mm_M256_N4096_K4096_cuda`
- `mm_M256_N4096_K512_cuda`
- `mm_M256_N512_K4096_cuda`
- `mm_M256_N512_K512_cuda`
- `mm_M3000_N4096_K512_cuda`
- `mm_M3000_N512_K4096_cuda`
- `mm_M3000_N512_K512_cuda`

### Compile Benchmark Unique Cases (73 unique)
- `add_M128_N32_K512_cuda`
- `add_M128_N64_K512_cuda`
- `add_M8_N32_K256_cuda`
- `add_M8_N64_K512_cuda`
- `addbmm_B32_M1024_N1024_K128_cuda`
- `addbmm_B32_M1024_N1024_K64_cuda`
- `addbmm_B32_M1024_N256_K128_cuda`
- `addbmm_B32_M1024_N256_K64_cuda`
- `addbmm_B32_M256_N1024_K128_cuda`
- `addbmm_B32_M256_N1024_K64_cuda`
- `addbmm_B32_M256_N256_K128_cuda`
- `addbmm_B32_M256_N256_K64_cuda`
- `addmm_M1024_N4096_K512_cuda`
- `addmm_M1024_N512_K4096_cuda`
- `addmm_M1024_N512_K512_cuda`
- `addmm_M256_N4096_K4096_cuda`
- `addmm_M256_N4096_K512_cuda`
- `addmm_M256_N512_K4096_cuda`
- `addmm_M256_N512_K512_cuda`
- `addmm_M3000_N4096_K4096_cuda`
- `addmm_M3000_N512_K512_cuda`
- `baddbmm_B32_M1024_N1024_K128_cuda`
- `baddbmm_B32_M1024_N1024_K64_cuda`
- `baddbmm_B32_M1024_N256_K128_cuda`
- `baddbmm_B32_M1024_N256_K64_cuda`
- `baddbmm_B32_M256_N1024_K128_cuda`
- `baddbmm_B32_M256_N1024_K64_cuda`
- `baddbmm_B32_M256_N256_K128_cuda`
- `baddbmm_B32_M256_N256_K64_cuda`
- `baddbmm_B8_M1024_N1024_K128_cuda`
- `baddbmm_B8_M1024_N1024_K64_cuda`
- `baddbmm_B8_M1024_N256_K128_cuda`
- `baddbmm_B8_M1024_N256_K64_cuda`
- `baddbmm_B8_M256_N1024_K128_cuda`
- `baddbmm_B8_M256_N256_K128_cuda`
- `baddbmm_B8_M256_N256_K64_cuda`
- `bmm_B32_M1024_N256_K128_cuda`
- `bmm_B32_M1024_N256_K64_cuda`
- `bmm_B32_M256_N1024_K128_cuda`
- `bmm_B32_M256_N1024_K64_cuda`
- `bmm_B32_M256_N256_K128_cuda`
- `bmm_B32_M256_N256_K64_cuda`
- `bmm_B8_M1024_N1024_K128_cuda`
- `bmm_B8_M1024_N1024_K64_cuda`
- `bmm_B8_M1024_N256_K128_cuda`
- `bmm_B8_M1024_N256_K64_cuda`
- `bmm_B8_M256_N1024_K128_cuda`
- `bmm_B8_M256_N1024_K64_cuda`
- `bmm_B8_M256_N256_K128_cuda`
- `bmm_B8_M256_N256_K64_cuda`
- `matmul_M1024_N4096_K4096`
- `matmul_M1024_N4096_K512`
- `matmul_M1024_N512_K4096`
- `matmul_M1024_N512_K512`
- `matmul_M256_N4096_K4096`
- `matmul_M256_N4096_K512`
- `matmul_M256_N512_K4096`
- `matmul_M256_N512_K512`
- `matmul_M3000_N4096_K4096`
- `matmul_M3000_N4096_K512`
- `matmul_M3000_N512_K4096`
- `matmul_M3000_N512_K512`
- `mm_M1024_N4096_K512_cuda`
- `mm_M1024_N512_K4096_cuda`
- `mm_M1024_N512_K512_cuda`
- `mm_M256_N4096_K4096_cuda`
- `mm_M256_N4096_K512_cuda`
- `mm_M256_N512_K4096_cuda`
- `mm_M256_N512_K512_cuda`
- `mm_M3000_N4096_K4096_cuda`
- `mm_M3000_N4096_K512_cuda`
- `mm_M3000_N512_K4096_cuda`
- `mm_M3000_N512_K512_cuda`

### All Unique Cases Combined (77 unique)
- `add_M128_N32_K512_cuda`
- `add_M128_N64_K512_cuda`
- `add_M8_N32_K256_cuda`
- `add_M8_N64_K512_cuda`
- `addbmm_B32_M1024_N1024_K128_cuda`
- `addbmm_B32_M1024_N1024_K64_cuda`
- `addbmm_B32_M1024_N256_K128_cuda`
- `addbmm_B32_M1024_N256_K64_cuda`
- `addbmm_B32_M256_N1024_K128_cuda`
- `addbmm_B32_M256_N1024_K64_cuda`
- `addbmm_B32_M256_N256_K128_cuda`
- `addbmm_B32_M256_N256_K64_cuda`
- `addmm_M1024_N4096_K4096_cuda`
- `addmm_M1024_N4096_K512_cuda`
- `addmm_M1024_N512_K4096_cuda`
- `addmm_M1024_N512_K512_cuda`
- `addmm_M256_N4096_K4096_cuda`
- `addmm_M256_N4096_K512_cuda`
- `addmm_M256_N512_K4096_cuda`
- `addmm_M256_N512_K512_cuda`
- `addmm_M3000_N4096_K4096_cuda`
- `addmm_M3000_N4096_K512_cuda`
- `addmm_M3000_N512_K4096_cuda`
- `addmm_M3000_N512_K512_cuda`
- `baddbmm_B32_M1024_N1024_K128_cuda`
- `baddbmm_B32_M1024_N1024_K64_cuda`
- `baddbmm_B32_M1024_N256_K128_cuda`
- `baddbmm_B32_M1024_N256_K64_cuda`
- `baddbmm_B32_M256_N1024_K128_cuda`
- `baddbmm_B32_M256_N1024_K64_cuda`
- `baddbmm_B32_M256_N256_K128_cuda`
- `baddbmm_B32_M256_N256_K64_cuda`
- `baddbmm_B8_M1024_N1024_K128_cuda`
- `baddbmm_B8_M1024_N1024_K64_cuda`
- `baddbmm_B8_M1024_N256_K128_cuda`
- `baddbmm_B8_M1024_N256_K64_cuda`
- `baddbmm_B8_M256_N1024_K128_cuda`
- `baddbmm_B8_M256_N1024_K64_cuda`
- `baddbmm_B8_M256_N256_K128_cuda`
- `baddbmm_B8_M256_N256_K64_cuda`
- `bmm_B32_M1024_N256_K128_cuda`
- `bmm_B32_M1024_N256_K64_cuda`
- `bmm_B32_M256_N1024_K128_cuda`
- `bmm_B32_M256_N1024_K64_cuda`
- `bmm_B32_M256_N256_K128_cuda`
- `bmm_B32_M256_N256_K64_cuda`
- `bmm_B8_M1024_N1024_K128_cuda`
- `bmm_B8_M1024_N1024_K64_cuda`
- `bmm_B8_M1024_N256_K128_cuda`
- `bmm_B8_M1024_N256_K64_cuda`
- `bmm_B8_M256_N1024_K128_cuda`
- `bmm_B8_M256_N1024_K64_cuda`
- `bmm_B8_M256_N256_K128_cuda`
- `bmm_B8_M256_N256_K64_cuda`
- `matmul_M1024_N4096_K4096`
- `matmul_M1024_N4096_K512`
- `matmul_M1024_N512_K4096`
- `matmul_M1024_N512_K512`
- `matmul_M256_N4096_K4096`
- `matmul_M256_N4096_K512`
- `matmul_M256_N512_K4096`
- `matmul_M256_N512_K512`
- `matmul_M3000_N4096_K4096`
- `matmul_M3000_N4096_K512`
- `matmul_M3000_N512_K4096`
- `matmul_M3000_N512_K512`
- `mm_M1024_N4096_K512_cuda`
- `mm_M1024_N512_K4096_cuda`
- `mm_M1024_N512_K512_cuda`
- `mm_M256_N4096_K4096_cuda`
- `mm_M256_N4096_K512_cuda`
- `mm_M256_N512_K4096_cuda`
- `mm_M256_N512_K512_cuda`
- `mm_M3000_N4096_K4096_cuda`
- `mm_M3000_N4096_K512_cuda`
- `mm_M3000_N512_K4096_cuda`
- `mm_M3000_N512_K512_cuda`

## Next Steps

1. Investigate the most severe regressions (>50% performance degradation)
2. Analyze patterns in backward pass operations
3. Review compile benchmark performance issues
4. Consider reverting or optimizing specific operations showing consistent regressions
5. Focus optimization efforts on the 77 unique cases identified above
