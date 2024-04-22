import numpy as np
import time

# Start time
start_time = time.time()

# Create a 10x10 matrix with random integers between 1 and 100
matrix = np.random.randint(1, 101, size=(10000, 10000))

# Print the matrix
# print("Original Matrix:")
# print(matrix)

# Calculate the transpose of the matrix
transpose_matrix = np.transpose(matrix)

# Print the transpose
# print("\nTransposed Matrix:")
# print(transpose_matrix)

# Calculate the sum of each column
column_sums = np.sum(matrix, axis=0)

# Print the column sums
# print("\nColumn Sums:")
# print(column_sums)

# End time
end_time = time.time()

# Calculate and print the execution time
execution_time = end_time - start_time
print("\nExecution Time:", execution_time, "seconds")
