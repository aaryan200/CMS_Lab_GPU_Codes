import cunumeric as np
import time

def array_to_string(arr):
    start_time = time.time()
    arr_str = np.array_str(arr)
    end_time = time.time()
    execution_time = end_time - start_time
    return arr_str, execution_time

# Example usage:
arr = np.random.rand(100, 100)
arr_str, exec_time = array_to_string(arr)
print("String representation of array:")
print(arr_str)
print("Execution time:", exec_time, "seconds")

