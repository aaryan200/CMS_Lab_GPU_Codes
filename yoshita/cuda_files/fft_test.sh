#!/bin/bash

# Compile the CUDA code
nvcc -o fft_cuda fft_cuda.cu -lcufft

# Define input sizes
sizes=("16 16" "32 32" "64 64" "128 128" "256 256" "512 512" "1024 1024")

# Loop over input sizes
for size in "${sizes[@]}"
do
    echo "Running FFT for size: $size"
    ./fft_cuda $size
    echo "---------------------------------------------"
donc

