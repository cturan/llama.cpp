# Cache Statistics Feature for llama.cpp

This document describes the cache statistics functionality added to llama.cpp for debugging and analyzing the recurrent cache behavior in models like Qwen3 Next.

## Overview

The cache statistics feature allows users to dump detailed information about the model's cache state after each token generation. This is particularly useful for:

- Understanding how the recurrent cache evolves during inference
- Debugging cache-related issues in hybrid models (attention + recurrent)
- Analyzing memory usage patterns
- Comparing cache behavior between different models

## Usage

### Command Line Option

Add the `--dump-cache` flag to any llama.cpp command to enable cache statistics printing:

```bash
./llama-cli -m your_model.gguf -p "Hello, my name is" -n 10 --dump-cache
```

### Test Script

A convenient test script is provided:

```bash
./test_cache_stats.sh /path/to/model.gguf "Your prompt here"
```

## Output Format

When enabled, the cache statistics are printed after each token generation:

```
=== CACHE STATISTICS FOR TOKEN 1 ===
Model has 32 layers
Memory address: 0x555555555555
Sequence 0: pos_min=0, pos_max=5, length=6
Memory supports shifting: true

Layer-by-layer cache information:
Note: Detailed tensor statistics require internal API access
This framework shows where conv/state/recurrent cache data would be displayed

Layer 0:
  Conv State: [sum=N/A, mean=N/A] (shape=N/A)
  Recurrent State: [sum=N/A, mean=N/A] (shape=N/A)
  Key Cache: [sum=N/A, mean=N/A] (shape=N/A)
  Value Cache: [sum=N/A, mean=N/A] (shape=N/A)

...

To access actual cache statistics, the following would be needed:
1. Internal API access to llama_memory_hybrid::get_mem_recr()
2. Access to llama_memory_recurrent::get_r_l() and ::get_s_l() tensors
3. Access to llama_kv_cache tensors for attention layers
4. ggml_tensor data access for sum/mean calculations
=============================================
```

## Implementation Details

### Files Modified

1. **tools/main/main.cpp**: Added cache statistics printing functionality
2. **common/common.h**: Added `dump_cache` parameter to `common_params` struct
3. **common/arg.cpp**: Added `--dump-cache` command line argument parsing

### Key Functions

- `print_cache_statistics()`: Main function that prints cache information
- Uses public llama.cpp APIs where available
- Provides framework for accessing internal cache data

### Limitations

The current implementation provides a framework for cache statistics but has limitations due to the public API constraints:

1. **Tensor Data Access**: Cannot directly access tensor data (sum, mean) without internal APIs
2. **Layer Type Detection**: Cannot distinguish between attention and recurrent layers
3. **Cache Type Identification**: Limited ability to determine specific cache types

### Future Enhancements

To fully implement cache statistics with actual tensor data, the following would be needed:

1. **Internal API Access**: Friend class access or new public APIs for cache internals
2. **Tensor Data Access**: Methods to access ggml_tensor data for calculations
3. **Layer Type Information**: APIs to determine layer types (attention vs recurrent)
4. **Cache Statistics Methods**: Built-in methods for cache statistics calculation

## Comparison with Python Reference

The Python reference implementation in `reference/tests/cache_stats_qwen3_next.py` provides full access to:

- Convolution state tensors (conv_states)
- Recurrent state tensors (recurrent_states)  
- Key/value cache tensors
- Actual sum and mean calculations

The C++ implementation aims to provide similar functionality once the necessary internal APIs are available.

## Troubleshooting

### No Cache Statistics Visible

If cache statistics don't appear:
1. Ensure `--dump-cache` flag is used
2. Check that the model supports cache operations
3. Verify the model is loaded correctly

### Memory Address Shows as Null

This indicates no memory is allocated for the cache, which could mean:
- Model doesn't support caching
- Memory allocation failed
- Incorrect model type

## Development Notes

For developers wanting to extend this functionality:

1. **Internal Access**: The main limitation is accessing internal cache structures
2. **API Design**: Consider adding public APIs for cache statistics
3. **Performance**: Cache statistics printing should have minimal performance impact
4. **Thread Safety**: Ensure thread safety when accessing cache data

## Related Files

- `reference/tests/cache_stats_qwen3_next.py`: Python reference implementation
- `src/llama-memory-hybrid.h`: Hybrid memory structure definitions
- `src/llama-memory-recurrent.h`: Recurrent memory structure definitions
- `src/llama-kv-cache.h`: KV cache structure definitions