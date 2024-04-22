import numpy as np
import time

n = int(input("Enter sixe of matrix:"))
# Generate a random array for testing
array_size = (n, n)  # Adjust the size as per your requirement
test_array = np.random.rand(*array_size)

# Measure execution time of np.fft.fft2
start_time = time.time()
result = np.fft.fft2(test_array)
end_time = time.time()

execution_time = end_time - start_time
print("Execution time of np.fft.fft2 using numpy:", execution_time, "seconds")