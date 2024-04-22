import cunumeric as np
from legate.timing import time

# Generate a random array for testing
array_size = (1000, 1000)  # Adjust the size as per your requirement
test_array = np.random.rand(*array_size)

# Measure execution time of np.fft.fft2
start_time = time()
result = np.fft.fft2(test_array)
end_time = time()

execution_time = end_time - start_time
print("Execution time of np.fft.fft2 using numpy:", execution_time, "seconds")
