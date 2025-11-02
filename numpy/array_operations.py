"""NumPy Array Operations - Basic Examples

This module demonstrates fundamental NumPy array operations including:
- Array creation and manipulation
- Indexing and slicing
- Mathematical operations
- Array reshaping
"""

import numpy as np

# Creating arrays
print("=== Array Creation ===")
# 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr_1d)

# 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr_2d)

# Arrays with specific values
zeros = np.zeros((3, 3))  # 3x3 array of zeros
ones = np.ones((2, 4))     # 2x4 array of ones
identity = np.eye(3)       # 3x3 identity matrix
print("\nZeros array:\n", zeros)
print("Ones array:\n", ones)
print("Identity matrix:\n", identity)

# Array with range
range_arr = np.arange(0, 10, 2)  # Start, stop, step
print("\nRange array:", range_arr)

# Linearly spaced array
linspace_arr = np.linspace(0, 1, 5)  # 5 values between 0 and 1
print("Linspace array:", linspace_arr)

# Basic Operations
print("\n=== Basic Operations ===")
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)
print("Power:", a ** 2)

# Array indexing and slicing
print("\n=== Indexing and Slicing ===")
arr = np.array([10, 20, 30, 40, 50])
print("Original array:", arr)
print("Element at index 2:", arr[2])
print("Slice [1:4]:", arr[1:4])
print("Every other element:", arr[::2])

# 2D array indexing
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Matrix:\n", matrix)
print("Element at [1, 2]:", matrix[1, 2])
print("First row:", matrix[0, :])
print("Second column:", matrix[:, 1])

# Array reshaping
print("\n=== Reshaping ===")
original = np.arange(12)
print("Original:", original)
reshaped = original.reshape(3, 4)
print("Reshaped to 3x4:\n", reshaped)
flattened = reshaped.flatten()
print("Flattened back:", flattened)

# Aggregation functions
print("\n=== Aggregation Functions ===")
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Data:", data)
print("Sum:", np.sum(data))
print("Mean:", np.mean(data))
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Standard deviation:", np.std(data))

# Boolean indexing
print("\n=== Boolean Indexing ===")
print("Elements > 5:", data[data > 5])
print("Even numbers:", data[data % 2 == 0])
