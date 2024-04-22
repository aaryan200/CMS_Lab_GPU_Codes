import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.fft as cu_fft

# Size of input array
NX = 1024

# Initialize input data (just an example)
input_data = np.arange(NX, dtype=np.complex64)

# Allocate memory on GPU
gpu_input = gpuarray.to_gpu(input_data)

# Create FFT plan
cu_fft.init()
plan = cu_fft.Plan(input_data.shape, np.complex64, np.complex64)

# Execute FFT
cu_fft.fft(gpu_input, gpu_input, plan)

# Clean up
cu_fft.shutdown()

print("FFT completed successfully.")

