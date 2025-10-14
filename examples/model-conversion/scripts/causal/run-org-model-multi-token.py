#!/usr/bin/env python3

import argparse
import os
import importlib
from pathlib import Path
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import numpy as np

### If you want to dump RoPE activations, apply this monkey patch to the model
### class from Transformers that you are running (replace apertus.modeling_apertus
### with the proper package and class for your model
### === START ROPE DEBUG ===
# from transformers.models.apertus.modeling_apertus import apply_rotary_pos_emb

# orig_rope = apply_rotary_pos_emb
# torch.set_printoptions(threshold=float('inf'))
# torch.set_printoptions(precision=6, sci_mode=False)

# def debug_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     # log inputs
#     summarize(q, "RoPE.q_in")
#     summarize(k, "RoPE.k_in")

#     # call original
#     q_out, k_out = orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)

#     # log outputs
#     summarize(q_out, "RoPE.q_out")
#     summarize(k_out, "RoPE.k_out")

#     return q_out, k_out

# # Patch it
# import transformers.models.apertus.modeling_apertus as apertus_mod  # noqa: E402
# apertus_mod.apply_rotary_pos_emb = debug_rope
### == END ROPE DEBUG ===

token_counter = {}

def summarize(tensor: torch.Tensor, name: str, max_seq: int = 3, max_vals: int = 3):
    global token, token_counter
    """
    Print a tensor in llama.cpp debug style.

    Supports:
    - 2D tensors (seq, hidden)
    - 3D tensors (batch, seq, hidden)
    - 4D tensors (batch, seq, heads, dim_per_head) via flattening heads × dim_per_head

    Shows first and last max_vals of each vector per sequence position.
    """
    t = tensor.detach().to(torch.float32).cpu()

    # Determine dimensions
    if t.ndim == 3:
        _, s, _ = t.shape
    elif t.ndim == 2:
        _, s = 1, t.shape[0]
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        _, s, _, _ = t.shape
    else:
        print(f"Skipping tensor due to unsupported dimensions: {t.ndim}")
        return

    ten_shape = t.shape

    print(f"ggml_debug: {name} = (f32)  ... = {{{ten_shape}}}")
    print("                                     [")
    print("                                      [")

    # Determine indices for first and last sequences
    first_indices = list(range(min(s, max_seq)))
    last_indices = list(range(max(0, s - max_seq), s))

    # Check if there's an overlap between first and last indices or if we're at the edge case of s = 2 * max_seq
    has_overlap = bool(set(first_indices) & set(last_indices)) or (max_seq * 2 == s)

    # Combine indices
    if has_overlap:
        # If there's overlap, just use the combined unique indices
        indices = sorted(list(set(first_indices + last_indices)))
        separator_index = None
    else:
        # If no overlap, we'll add a separator between first and last sequences
        indices = first_indices + last_indices
        separator_index = len(first_indices)

    for i, si in enumerate(indices):
        # Add separator if needed
        if separator_index is not None and i == separator_index:
            print("                                       ...")

        # Extract appropriate slice
        vec = t[0, si]
        if vec.ndim == 2:  # 4D case: flatten heads × dim_per_head
            flat = vec.flatten().tolist()
        else:  # 2D or 3D case
            flat = vec.tolist()

        # First and last slices
        first = flat[:max_vals]
        last = flat[-max_vals:] if len(flat) >= 2 * max_vals else flat
        first_str = ", ".join(f"{v:12.4f}" for v in first)
        last_str = ", ".join(f"{v:12.4f}" for v in last)

        if len(flat) >= 2 * max_vals:
            print(f"                                       [{first_str}, ..., {last_str}]")
        else:
            print(f"                                       [{last_str}]")

    print("                                      ],")
    print("                                     ]")
    print(f"                                     sum = {t.sum().item():.6f}\n")

    pattern = r"model\.layers\.[0-9]+_out"
    pattern2 = r"recurrent_cache_[0-9]+"
    if re.fullmatch(pattern, name) or re.fullmatch(pattern2, name):
        if name not in token_counter:
            token_counter[name] = 1
        else:
            token_counter[name] = token_counter[name] + 1
        save_tensor(t, f"reference/tensors/org/{name}_{token_counter[name]}.bin")

from transformers.models.qwen3_next.modeling_qwen3_next import torch_causal_conv1d_update, apply_rotary_pos_emb, l2norm  # noqa: E402
orig_conv1d_update = torch_causal_conv1d_update
orig_rope = apply_rotary_pos_emb
import torch.nn.functional as F  # noqa: E402
import typing  # noqa: E402

def patched_torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    summarize(hidden_states, "hidden_states_in")
    summarize(conv_state, "conv_state_in")

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    summarize(hidden_states_new, "hidden_states_new")
    summarize(hidden_states_new[:, :, -state_len:], "hidden_states_to_copy")
    summarize(conv_state, "conv_state_pre")
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    summarize(conv_state, "conv_state_post")
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    summarize(out, "out")
    summarize(out[:, :, -seq_len:], "out_proper")
    out = F.silu(out[:, :, -seq_len:])
    summarize(out, "out_silu")
    out = out.to(hidden_states.dtype)
    return out

already_dumped_rope = False

def save_tensor(tensor, filename):
    """Save tensor to binary file with shape information."""
    # Ensure tensors directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert to numpy and save
    np_array = tensor.detach().cpu().numpy()
    
    # Save shape first (4 int64 values), then data
    with open(filename, 'wb') as f:
        shape = list(np_array.shape)
        while len(shape) < 4:
            shape.insert(0, 0)
        
        # Write shape as int64
        shape_array = np.array(shape, dtype=np.int64)
        f.write(shape_array.tobytes())
        
        # Write data as float32
        np_array_float32 = np_array.astype(np.float32)
        f.write(np_array_float32.tobytes())

def patched_apply_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    global already_dumped_rope

    # log inputs
    summarize(q, "RoPE.q_in")
    summarize(k, "RoPE.k_in")
    summarize(cos, "cos")
    summarize(sin, "sin")
    # if q.shape[1] == 2 and k.shape[1] == 1 and k.shape[2] == 1 and not already_dumped_rope:
    #     already_dumped_rope = True
    #     print("Dumping input tensors")
    #     save_tensor(q, "reference/tensors/testrope_q_in.bin")
    #     save_tensor(k, "reference/tensors/testrope_k_in.bin")
    #     save_tensor(cos, "reference/tensors/testrope_cos_in.bin")
    #     save_tensor(sin, "reference/tensors/testrope_sin_in.bin")

    if position_ids:
        summarize(position_ids, "position_ids")
    # print(f"Rotary dim is {cos.unsqueeze(unsqueeze_dim).shape[-1]}")

    # call original
    q_out, k_out = orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)

    # log outputs
    summarize(q_out, "RoPE.q_out")
    summarize(k_out, "RoPE.k_out")

    return q_out, k_out

def patched_torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    long=False
):
    torch.set_printoptions(threshold=10_000_000, sci_mode=False, precision=10, linewidth=200)
    initial_dtype = query.dtype
    [ summarize(x, y) for (x, y) in ((query, "q_prenorm"), (key, "k_prenorm")) ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    [ summarize(x, y) for (x, y) in ((query, "q_orig"), (key, "k_orig"), (value, "v_orig"), (beta, "b_orig"), (g, "g_orig")) ]
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    [ summarize(x, y) for (x, y) in ((query, "q_tra"), (key, "k_tra"), (value, "v_tra"), (beta, "b_tra"), (g, "g_tra")) ]
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    print(f"batch_size = {batch_size}, seq_len = {sequence_length}, num_heads = {num_heads}, k_head_dim = {k_head_dim}")
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    print(f"Pad size = {pad_size}, chunk_size = {chunk_size}")
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    [ summarize(x, y) for (x, y) in ((query, "q_pad"), (key, "k_pad"), (value, "v_pad"), (beta, "b_pad"), (g, "g_pad")) ]
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    print(f"Scale for delta is {scale} (from {query.shape[-1]})")
    query = query * scale

    summarize(query, "q_scaled")
    summarize(key, "k")
    summarize(beta.unsqueeze(-1), "beta")
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    summarize(k_beta, "k_beta")
    summarize(v_beta, "v_beta")
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    [ summarize(x, y) for (x, y) in ((query, "q_resh"), (k_beta, "k_beta_resh"), (v_beta, "v_beta_resh"), (key, "k_resh"), (value, "v_resh")) ]

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    summarize(g, "g_cumsum")
    sub = g.unsqueeze(-1) - g.unsqueeze(-2)
    bt1, bt2 = torch.broadcast_tensors(g.unsqueeze(-1), g.unsqueeze(-2))
    summarize(bt1, "bt1")
    summarize(bt2, "bt2")
    summarize(sub, "sub")
    decay_mask = sub.tril()
    summarize(decay_mask, "sub_tril")
    decay_mask = decay_mask.exp()
    summarize(decay_mask, "sub_tril_exp")
    decay_mask = decay_mask.float()
    summarize(decay_mask, "sub_tril_exp_float")
    decay_mask = decay_mask.tril()
    summarize(decay_mask, "decay_mask")
    k_t = key.transpose(-1, -2)
    summarize(k_t, "k_t")
    kmul = k_beta @ k_t
    summarize(kmul, "k_beta @ k_t")
    #if not long:
        #print(f"k_beta @ k_t:\n{kmul[:,:,:,:8,:8]}\n\n")
    kmul_decay = kmul * decay_mask
    summarize(kmul_decay, "(k_beta @ k_t) * decay_mask")
    attn = -(kmul_decay).masked_fill(mask, 0)
    summarize(attn, "attn_in")
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
        #if i <= num_heads and not long: 
            #print(f"Chunk {i}: row:\n{row}\n\nsub:\n{sub}\nrow_unsq:\n{row.unsqueeze(-1)}\nrow_unsq * sub:\n{row.unsqueeze(-1)*sub}\n")
            #print(f"attn => sum = {attn[..., i, :i].sum()}, tensor: \n{attn[..., i, :i]}\n\n")
    summarize(attn, "attn_chunks")
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    summarize(attn, "attn_eye")
    
    value = attn @ v_beta
    summarize(value, "value")
        
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    summarize(k_cumdecay, "k_cumdecay")
    
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        print(f"\n=== Processing chunk {i} ===")
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        summarize(q_i, f"q_i_chunk_{i}")
        summarize(k_i, f"k_i_chunk_{i}")
        summarize(v_i, f"v_i_chunk_{i}")
        
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        summarize(attn, f"attn_chunk_{i}")
        
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        summarize(v_prime, f"v_prime_chunk_{i}")
        
        v_new = v_i - v_prime
        summarize(v_new, f"v_new_chunk_{i}")
        
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        summarize(attn_inter, f"attn_inter_chunk_{i}")
        
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        summarize(core_attn_out[:, :, i], f"core_attn_out_chunk_{i}")
        
        g_last = g[:, :, i, -1, None, None].exp()
        summarize(g_last, f"g_last_chunk_{i}")
        
        g_diff_exp = (g[:, :, i, -1, None] - g[:, :, i]).exp()
        last_recurrent_state = (
            last_recurrent_state * g_last
            + (k_i * g_diff_exp[..., None]).transpose(-1, -2) @ v_new
        )
        summarize(last_recurrent_state, f"updated_state_chunk_{i}")

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    summarize(core_attn_out, "attn_out")
    if not long:
        print(f"attn_out:\n{core_attn_out}\n\n")
        
    if isinstance(last_recurrent_state, torch.Tensor):
        summarize(last_recurrent_state, "state_out")
        if not long:
            print(f"state_out:\n{last_recurrent_state}\n\n")
    return core_attn_out, last_recurrent_state


def patched_torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    summarize(query, "q_t")
    summarize(key, "k_t")
    summarize(value, "v_t")
    summarize(beta, "beta_t")
    summarize(g, "g_t")

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    summarize(query, "q_scaled")
    if initial_state is not None:
        summarize(initial_state, "initial_state")

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        summarize(g_t, "g_exp_unsq")
        beta_t = beta[:, :, i].unsqueeze(-1)
        summarize(beta_t, "beta_t_unsq")

        last_recurrent_state = last_recurrent_state * g_t
        summarize(last_recurrent_state, "gated_state")
        k_unsq = k_t.unsqueeze(-1)
        summarize(k_unsq, "k_unsqueeze")
        state_k = last_recurrent_state * k_unsq
        summarize(state_k, "state_k_product")
        kv_mem = state_k.sum(dim=-2)
        summarize(kv_mem, "kv_mem")
        delta = (v_t - kv_mem) * beta_t
        summarize(delta, "delta")
        k_delta = k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        summarize(k_delta, "k_delta")
        last_recurrent_state = last_recurrent_state + k_delta
        summarize(last_recurrent_state, "state_plus_k_delta")
        state_q_prod = last_recurrent_state * q_t.unsqueeze(-1)
        summarize(state_q_prod, "state_q_product")
        core_attn_out[:, :, i] = state_q_prod.sum(dim=-2)
        summarize(core_attn_out, "core_attn_out")

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

import transformers.models.qwen3_next.modeling_qwen3_next as qwen_mod  # noqa: E402
qwen_mod.torch_chunk_gated_delta_rule = patched_torch_chunk_gated_delta_rule
qwen_mod.torch_causal_conv1d_update = patched_torch_causal_conv1d_update
qwen_mod.apply_rotary_pos_emb = patched_apply_rope
qwen_mod.torch_recurrent_gated_delta_rule = patched_torch_recurrent_gated_delta_rule

# Store original functions for patching
original_functions = {}

def debug_hook(name):
    def fn(_m, input, output):
        if isinstance(input, torch.Tensor):
            summarize(input, name + "_in")
        elif isinstance(input, (tuple, list)) and len(input) > 0 and isinstance(input[0], torch.Tensor):
            summarize(input[0], name + "_in")
        if isinstance(output, torch.Tensor):
            summarize(output, name + "_out")
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            summarize(output[0], name + "_out")

    return fn

def patch_all_forward_methods(model):
    """Apply monkey patches to all forward methods in the model"""
    for name, module in model.named_modules():
        # Set layer index if applicable
        parts = name.split('.')
        module.layer_idx = -1 # Default invalid value

        if len(parts) > 2 and parts[0] == 'model' and parts[1] == 'layers':
            try:
                module.layer_idx = int(parts[2])  # Convert to integer
            except (ValueError, IndexError):
                module.layer_idx = -1

        # Apply forward hook to log all inputs/outputs
        module.register_forward_hook(debug_hook(name))

        # Additional patches for specific methods in various modules
        if hasattr(module, 'forward'):
            original_forward = module.forward
            def make_patched_forward(orig_forward, mod_name):
                def patched_forward(*args, **kwargs):
                    # Log inputs
                    for i, arg in enumerate(args):
                        if isinstance(arg, torch.Tensor):
                            summarize(arg, f"{mod_name}.forward.arg_{i}_in")

                    # Call original forward
                    result = orig_forward(*args, **kwargs)

                    if mod_name.endswith("linear_attn"):
                        cache = kwargs["cache_params"]
                        nameparts = mod_name.split(".")
                        layer_idx = -1
                        try:
                            layer_idx = int(nameparts[2])
                        except (ValueError, IndexError):
                            print(f"\n\nDEBUG: Failed to calculate layer index for module: {mod_name}\n\n")
                        rec_cache = cache.recurrent_states[layer_idx]
                        if rec_cache is not None:
                            summarize(rec_cache, f"recurrent_cache_{layer_idx}")

                    # Log output
                    if isinstance(result, torch.Tensor):
                        summarize(result, f"{mod_name}.forward.out")
                    elif isinstance(result, (tuple, list)):
                        for i, res in enumerate(result):
                            if isinstance(res, torch.Tensor):
                                summarize(res, f"{mod_name}.forward.out_{i}")

                    return result
                return patched_forward

            module.forward = make_patched_forward(original_forward, name)

def patch_silu():
    """Patch torch.nn.functional.silu to log inputs and outputs"""
    global original_functions

    if 'silu' not in original_functions:
        original_functions['silu'] = torch.nn.functional.silu

    def patched_silu(input, inplace=False):
        # Log input
        summarize(input, "silu_in")

        # Call original function
        result = original_functions['silu'](input, inplace)

        # Log output
        summarize(result, "silu_out")

        return result

    # Replace the function in the torch.nn.functional module
    torch.nn.functional.silu = patched_silu

def patch_sigmoid():
    """Patch torch.nn.functional.sigmoid to log inputs and outputs"""
    global original_functions

    if 'sigmoid' not in original_functions:
        original_functions['sigmoid'] = torch.nn.functional.sigmoid

    def patched_sigmoid(input):
        # Log input
        summarize(input, "sigmoid_in")

        # Call original function
        result = original_functions['sigmoid'](input)

        # Log output
        summarize(result, "sigmoid_out")

        return result

    # Replace the function in the torch.nn.functional module
    torch.nn.functional.sigmoid = patched_sigmoid


def patch_torch_sigmoid():
    """Patch torch.nn.functional.sigmoid to log inputs and outputs"""
    global original_functions

    if 'torch_sigmoid' not in original_functions:
        original_functions['torch_sigmoid'] = torch.sigmoid

    def patched_torch_sigmoid(input):
        # Log input
        summarize(input, "torch_sigmoid_in")

        # Call original function
        result = original_functions['torch_sigmoid'](input)

        # Log output
        summarize(result, "torch_sigmoid_out")

        return result

    # Replace the function in the torch.nn.functional module
    torch.sigmoid = patched_torch_sigmoid


def patch_pad():
    """Patch torch.nn.functional.pad to log inputs and outputs"""
    global original_functions

    if 'pad' not in original_functions:
        original_functions['pad'] = torch.nn.functional.pad

    def patched_pad(input: torch.Tensor, pad: typing.Sequence[int], mode: str = 'constant', value: float | None = None): # pyright: ignore[reportGeneralTypeIssues]
        # Log input
        summarize(input, "pad_in")
        print(f"Padding shape is {pad}")

        # Call original function
        result = original_functions['pad'](input=input, pad=pad, mode=mode, value=value)

        # Log output
        summarize(result, "pad_out")

        return result

    # Replace the function in the torch.nn.functional module
    torch.nn.functional.pad = patched_pad


def save_kv_cache(past_key_values, step_num, data_dir, model_name):
    """Save KV cache tensors for each layer"""
    cache_dir = data_dir / f"kv_cache_step_{step_num}"
    cache_dir.mkdir(exist_ok=True)
    
    # Access past_key_values if available
    if past_key_values is not None:
        for layer_idx, cache_tuple in enumerate(past_key_values):
            if cache_tuple is None:
                print(f"Cache tuple is None for layer {layer_idx} at step {step_num}")
                continue
                
            # Handle different cache formats
            if isinstance(cache_tuple, (tuple, list)) and len(cache_tuple) >= 2:
                key, value = cache_tuple[0], cache_tuple[1]
                
                # Check if key and value are not None
                if key is not None and value is not None:
                    # Save key cache
                    key_filename = cache_dir / f"layer_{layer_idx}_key.bin"
                    key.detach().cpu().numpy().astype(np.float32).tofile(key_filename)
                    
                    # Save value cache
                    value_filename = cache_dir / f"layer_{layer_idx}_value.bin"
                    value.detach().cpu().numpy().astype(np.float32).tofile(value_filename)
                    
                    print(f"Saved KV cache for layer {layer_idx} at step {step_num}: key.shape={key.shape}, value.shape={value.shape}")
                else:
                    print(f"Key or value is None for layer {layer_idx} at step {step_num}")
            else:
                # Handle other cache formats (e.g., recurrent models)
                print(f"Non-standard cache format for layer {layer_idx} at step {step_num}: {type(cache_tuple)}")
                # Save as generic cache if it's a tensor
                if hasattr(cache_tuple, 'detach'):
                    cache_filename = cache_dir / f"layer_{layer_idx}_cache.bin"
                    cache_tuple.detach().cpu().numpy().astype(np.float32).tofile(cache_filename)
                    print(f"Saved generic cache for layer {layer_idx} at step {step_num}: shape={cache_tuple.shape}")
    else:
        print(f"No KV cache available at step {step_num}")


unreleased_model_name = os.getenv("UNRELEASED_MODEL_NAME")

parser = argparse.ArgumentParser(description="Process model with specified path")
parser.add_argument("--model-path", "-m", help="Path to the model")
parser.add_argument("--num-tokens", "-n", type=int, default=5, help="Number of tokens to generate")
parser.add_argument("--prompt", "-p", default="Hello, my name is", help="Input prompt")
parser.add_argument("--save-cache", action="store_true", help="Save KV cache at each step")
args = parser.parse_args()

model_path = os.environ.get("MODEL_PATH", args.model_path)
if model_path is None:
    parser.error(
        "Model path must be specified either via --model-path argument or MODEL_PATH environment variable"
    )

config = AutoConfig.from_pretrained(model_path)

print("Model type:       ", config.model_type)
print("Vocab size:       ", config.vocab_size)
print("Hidden size:      ", config.hidden_size)
print("Number of layers: ", config.num_hidden_layers)
print("BOS token id:     ", config.bos_token_id)
print("EOS token id:     ", config.eos_token_id)

print("Loading model and tokenizer using AutoTokenizer:", model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)

if unreleased_model_name:
    model_name_lower = unreleased_model_name.lower()
    unreleased_module_path = (
        f"transformers.models.{model_name_lower}.modular_{model_name_lower}"
    )
    class_name = f"{unreleased_model_name}ForCausalLM"
    print(f"Importing unreleased model module: {unreleased_module_path}")

    try:
        model_class = getattr(
            importlib.import_module(unreleased_module_path), class_name
        )
        model = model_class.from_pretrained(
            model_path
        )  # Note: from_pretrained, not fromPretrained
    except (ImportError, AttributeError) as e:
        print(f"Failed to import or load model: {e}")
        exit(1)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", offload_folder="offload"
    )

patch_all_forward_methods(model)
patch_silu()
patch_pad()
patch_sigmoid()
patch_torch_sigmoid()

model_name = os.path.basename(model_path)
# Printing the Model class to allow for easier debugging. This can be useful
# when working with models that have not been publicly released yet and this
# migth require that the concrete class is imported and used directly instead
# of using AutoModelForCausalLM.
print(f"Model class: {model.__class__.__name__}")

device = next(model.parameters()).device
prompt = args.prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

print(f"Input tokens: {input_ids}")
print(f"Input text: {repr(prompt)}")
print(f"Tokenized: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Store all generated tokens and logits
all_generated_tokens = []
all_logits = []

with torch.no_grad():
    # Initial forward pass
    print(f"\n=== Initial Forward Pass ===")
    outputs = model(input_ids, use_cache=True)
    logits = outputs.logits
    
    # Extract logits for the last token (next token prediction)
    last_logits = logits[0, -1, :].cpu().numpy()
    all_logits.append(last_logits)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Last token logits shape: {last_logits.shape}")
    
    # Generate first token
    next_token_id = np.argmax(last_logits).item()
    all_generated_tokens.append(next_token_id)
    
    # Show top 5 predicted tokens for first step
    top_indices = np.argsort(last_logits)[-5:][::-1]
    print("Top 5 predictions for first token:")
    for idx in top_indices:
        token = tokenizer.decode([idx])
        print(f"  Token {idx} ({repr(token)}): {last_logits[idx]:.6f}")
    
    print(f"Generated token {next_token_id} ({repr(tokenizer.decode([next_token_id]))})")
    
    # Save KV cache if requested
    if args.save_cache:
        save_kv_cache(outputs.past_key_values, 0, data_dir, model_name)
    
    # Prepare for next iteration
    past_key_values = outputs.past_key_values
    current_input = torch.tensor([[next_token_id]], device=device)
    
    # Generate remaining tokens
    for step in range(1, args.num_tokens):
        print(f"\n=== Generation Step {step} ===")
        
        # Forward pass with cache
        outputs = model(
            input_ids=current_input, 
            past_key_values=past_key_values,
            use_cache=True
        )
        
        logits = outputs.logits
        last_logits = logits[0, -1, :].cpu().numpy()
        all_logits.append(last_logits)
        
        # Generate next token
        next_token_id = np.argmax(last_logits).item()
        all_generated_tokens.append(next_token_id)
        
        # Show top 5 predicted tokens for this step
        top_indices = np.argsort(last_logits)[-5:][::-1]
        print(f"Top 5 predictions for step {step}:")
        for idx in top_indices:
            token = tokenizer.decode([idx])
            print(f"  Token {idx} ({repr(token)}): {last_logits[idx]:.6f}")
        
        print(f"Generated token {next_token_id} ({repr(tokenizer.decode([next_token_id]))})")
        
        # Save KV cache if requested
        if args.save_cache:
            save_kv_cache(outputs.past_key_values, step, data_dir, model_name)
        
        # Update for next iteration
        past_key_values = outputs.past_key_values
        current_input = torch.tensor([[next_token_id]], device=device)

# Save results
bin_filename = data_dir / f"pytorch-{model_name}-multi-token.bin"
txt_filename = data_dir / f"pytorch-{model_name}-multi-token.txt"

# Save all logits concatenated
all_logits_array = np.array(all_logits)
all_logits_array.astype(np.float32).tofile(bin_filename)

# Also save as text file for easy inspection
with open(txt_filename, "w") as f:
    f.write(f"Generated tokens: {all_generated_tokens}\n")
    f.write(f"Generated text: {repr(tokenizer.decode(all_generated_tokens))}\n")
    f.write(f"Full sequence: {repr(tokenizer.decode(input_ids[0].tolist() + all_generated_tokens))}\n\n")
    
    for step, logits in enumerate(all_logits):
        f.write(f"=== Step {step} logits ===\n")
        for i, logit in enumerate(logits):
            f.write(f"{i}: {logit:.6f}\n")
        f.write("\n")

print(f"\n=== Generation Complete ===")
print(f"Generated {len(all_generated_tokens)} tokens: {all_generated_tokens}")
print(f"Generated text: {repr(tokenizer.decode(all_generated_tokens))}")
print(f"Full sequence: {repr(tokenizer.decode(input_ids[0].tolist() + all_generated_tokens))}")

print(f"Saved bin logits to: {bin_filename}")
print(f"Saved txt logits to: {txt_filename}")

if args.save_cache:
    print(f"KV cache saved to: {data_dir}/kv_cache_step_*")