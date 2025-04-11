#!/usr/bin/env python
# coding=utf-8
import argparse
from cvxopt.glpk import ilp
import numpy as np
from cvxopt import matrix
import torch
import pickle
import math

def solve_gpu_split(
    activation_path: str,
    neuron: int,
    capacity: int,
    layer: int,
    batch: int,
    threshold: int,
    computility_ratio_up: float,
    computility_ratio_down: float,
    has_gate: bool
):
    result = [[0 for j in range(2)] for i in range(layer)]

    # average solve
    # average = math.floor(capacity / layer)
    # for i in range(layer):
    #     result[i][0] = average
    #     result[i][1] = average
    # return result

    # Processing activation data
    freqs = []
    sums = []
    for i in range(layer):
        # Load and sort activation data for each layer
        freq = torch.load(f"{activation_path}/activation_{i}.pt")
        freq, _ = torch.sort(freq, descending=True)
        freq = freq.view(-1, batch)
        freq = freq.sum(dim=1)
        freq = freq.tolist()
        freqs.append(freq)
        sums.append(sum(freq))
    
    batch_num = math.ceil(neuron / batch)
    cached_freq_sum = [[0 for j in range(2)] for i in range(layer)]

    for i in range(batch_num):
        for j in range(layer):
            # up
            up_unit = batch * 2 if has_gate else batch
            up_cached_freq = cached_freq_sum[j][0]
            if capacity >= up_unit and up_cached_freq / (sums[j] - up_cached_freq) < computility_ratio_up:
                result[j][0] += batch
                cached_freq_sum[j][0] += freqs[j][i]
                capacity -= up_unit
                
            # down
            down_unit = batch
            down_cached_freq = cached_freq_sum[j][1]
            if capacity >= down_unit and down_cached_freq / (sums[j] - down_cached_freq) < computility_ratio_down:
                result[j][1] += batch
                cached_freq_sum[j][1] += freqs[j][i]
                capacity -= down_unit
        
        if capacity < batch:
            break
            
    return result

def compute_inference_time(fc_ratio, attn_ratio, cpu_capacity, gpu_capacity, memory_size):
    cpu_time = (1 - fc_ratio) * cpu_capacity + (1 - attn_ratio) * cpu_capacity
    gpu_time = fc_ratio * gpu_capacity + attn_ratio * gpu_capacity
    memory_penalty = max(0, fc_ratio + attn_ratio - memory_size)
    return cpu_time + gpu_time + memory_penalty

def compute_energy_consumption(fc_ratio, attn_ratio, cpu_energy, gpu_energy):
    cpu_energy_consumption = (1 - fc_ratio) * cpu_energy + (1 - attn_ratio) * cpu_energy
    gpu_energy_consumption = fc_ratio * gpu_energy + attn_ratio * gpu_energy
    return cpu_energy_consumption + gpu_energy_consumption

def compute_gradient(inference_time, energy_consumption, fc_ratio, attn_ratio):
    fc_ratio_tensor = torch.tensor(fc_ratio, requires_grad=True)
    attn_ratio_tensor = torch.tensor(attn_ratio, requires_grad=True)
    
    objective = inference_time + energy_consumption
    
    objective.backward()
    
    return fc_ratio_tensor.grad.item(), attn_ratio_tensor.grad.item()

def optimize_split_ratio(cpu_capacity, gpu_capacity, memory_size, cpu_energy, gpu_energy, learning_rate, max_iter):
    fc_ratio = 0.5
    attn_ratio = 0.5

    for _ in range(max_iter):
        inference_time = compute_inference_time(fc_ratio, attn_ratio, cpu_capacity, gpu_capacity, memory_size)
        energy_consumption = compute_energy_consumption(fc_ratio, attn_ratio, cpu_energy, gpu_energy)

        fc_gradient, attn_gradient = compute_gradient(inference_time, energy_consumption, fc_ratio, attn_ratio)

        fc_ratio -= learning_rate * fc_gradient
        attn_ratio -= learning_rate * attn_gradient
        

        fc_ratio = max(0, min(1, fc_ratio))
        attn_ratio = max(0, min(1, attn_ratio))
    
    return fc_ratio, attn_ratio