import quantize
import math
import torch
import pytest
import numpy as np

from common import iter_indices

torch.Tensor.quantize_ = quantize.quantize_
torch.Tensor.saturate_ = quantize.saturate_
torch.cuda.FloatTensor.quantize_ = quantize.quantize_
torch.cuda.FloatTensor.saturate_ = quantize.saturate_

def check_saturation(m1, scale_factor, bits):
	min_val = -scale_factor*math.pow(2, bits-1)
	max_val = scale_factor*(math.pow(2, bits-1) - 1)
	m2 = m1.clone()
	for i in iter_indices(m2):
	    m2[i] = max(min_val, min(max_val, m2[i]))
	np.testing.assert_equal(m1.numpy(), m2.numpy())

def check_quantization(m1, scale_factor, bits):
	# Test that quantized value is in it's range
	check_saturation(m1, scale_factor, bits)

	# Test that quantized value is representable
	m2 = m1.clone()
	for i in iter_indices(m2):
		# Must be an integer in the fixed-point representation
		m2[i] = round(m2[i] / scale_factor) * scale_factor
	np.testing.assert_allclose(m1.numpy(), m2.numpy(), rtol=1e-6)

@pytest.mark.parametrize("scale_factor,bits",
[
    (0.05, 8),
    (5e-5, 16),
    (5e-9, 32)
])
def test_quantize(scale_factor, bits):
	# Create a matrix with 100 values randomly uniform within [-15, 15]
	m1 = torch.rand(100).mul(30).add(-15)
	m1.quantize_(scale_factor, bits)
	check_quantization(m1, scale_factor, bits)

@pytest.mark.parametrize("scale_factor,bits",
[
    (0.05, 8),
    (5e-5, 16),
    (5e-9, 32)
])
def test_saturate(scale_factor, bits):
	m1 = torch.rand(100).mul(30).add(-15) # uniform [-15, 15]
	m1.saturate_(scale_factor, bits)
	# Test that saturated value is in it's range
	check_saturation(m1, scale_factor, bits)