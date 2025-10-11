#!/bin/bash

# Test script for cache statistics functionality
# This script demonstrates how to use the --dump-cache flag

echo "Testing llama.cpp cache statistics functionality"
echo "=============================================="

# Check if a model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_model.gguf> [prompt]"
    echo "Example: $0 /path/to/qwen3-next.gguf \"Hello, my name is\""
    exit 1
fi

MODEL_PATH="$1"
PROMPT="${2:-Hello, my name is}"

echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo ""

# Run llama.cpp with cache statistics enabled
echo "Running: ./llama-cli -m $MODEL_PATH -p \"$PROMPT\" -n 5 --dump-cache"
echo ""

# Build the command
CMD="./build/bin/llama-cli -m $MODEL_PATH -p \"$PROMPT\" -n 5 --dump-cache"

# Execute the command
echo "Executing: $CMD"
echo ""
eval $CMD

echo ""
echo "Cache statistics test completed."
echo "=============================================="