import math
import torch

# Modified from
# https://github.com/aaron-xichen/pytorch-playground/blob/master/utee/quant.py

def quantize_(input, scale_factor, bits, biased=False):
    assert bits >= 1, bits
    bound = math.pow(2.0, bits-1)
    min_val = - bound
    max_val = bound - 1
    if biased:
    	adj_val = 0.5
    else:
    	# Generate tensor of random values from [0,1]
    	adj_val = torch.Tensor(input.size()).type(input.type()).uniform_()

    rounded = input.div_(scale_factor).add_(adj_val).floor_()
    clipped_value = rounded.clamp_(min_val, max_val)
    clipped_value *= scale_factor

def saturate_(input, scale_factor, bits):
	bound = math.pow(2.0, bits-1)
	min_val = - bound * scale_factor
	max_val = (bound-1) * scale_factor
	input.clamp_(min_val, max_val)

# Monkey patch torch.Tensor
torch.Tensor.quantize_ = quantize_
torch.Tensor.saturate_ = saturate_
torch.cuda.FloatTensor.quantize_ = quantize_
torch.cuda.FloatTensor.saturate_ = saturate_