import numpy as np
import time
import matplotlib.pyplot as plt

def run_fft_with_timing(matrix_size):
    # Generate a random array for testing
    array_size = (matrix_size, matrix_size)
    test_array = np.random.rand(*array_size)

    # Measure execution time of np.fft.fft2
    start_time = time.time()
    result = np.fft.fft2(test_array)
    end_time = time.time()
    # print(f"Size={matrix_size}")
    # print(result[0,0])
    # print("C\n")

    execution_time = end_time - start_time
    return execution_time

# List of matrix sizes to test
matrix_sizes = [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]

# Run FFT with timing for each matrix size
execution_times = [run_fft_with_timing(size) for size in matrix_sizes]

# Plotting
plt.plot(matrix_sizes, execution_times, marker='o', linestyle='-',label='Numpy')

import cunumeric as np
# Run FFT with timing for each matrix size for cunumeric
execution_times = [run_fft_with_timing(size) for size in matrix_sizes]

# Plotting
plt.plot(matrix_sizes, execution_times, marker='o', linestyle='-',label='CuNumeric')


plt.title('Execution Time vs Matrix Size')
plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.legend()
plt.show()
