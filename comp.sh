#!/bin/bash
echo "Running converted model."
llama-cli -no-cnv -m reference/qwen3_ntl/qwen3_ntl.gguf -p "Once upon a time" -n 30 --temp 0 &> data/tinylong-30-tok.txt
echo "Running original model."
python examples/model-conversion/scripts/causal/run-org-model-multi-token.py --model-path reference/qwen3_ntl --num-tokens 30 --prompt "Once upon a time" &> data/tinylong-30-tok-org.txt
echo "Running tensor comparison."
python reference/compare_tensors.py 30 16 &> data/tinylong-30-compare.txt
echo "Done."