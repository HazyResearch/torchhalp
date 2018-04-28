# Linear Regression Example

To run linear regression for HALP:

    python main.py --halp --epochs 50 --no-cuda --save_graph

Note: if you want to reproduce the results in the paper, we additionally quantized the input for all optimizers for linear regression since this could speed up our C++ implementation. We have found that the quantized input tends to (negatively) affect the performance of full-precision SGD more than full-precision SVRG. Here, we leave the input unquantized.
