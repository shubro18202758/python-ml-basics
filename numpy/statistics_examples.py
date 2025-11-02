"""NumPy Statistics Examples

This module demonstrates NumPy statistical functions including:
- Descriptive statistics (mean, median, mode, variance, std)
- Distributions and random number generation
- Correlation and covariance
- Percentiles and quantiles
"""

import numpy as np

# Setting random seed for reproducibility
np.random.seed(42)

# Basic descriptive statistics
print("=== Basic Descriptive Statistics ===")
data = np.array([12, 15, 18, 21, 24, 27, 30, 33, 36, 39])

print("Data:", data)
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))
print("Min:", np.min(data))
print("Max:", np.max(data))
print("Range:", np.ptp(data))  # Peak to peak (max - min)

# Percentiles and quantiles
print("\n=== Percentiles and Quantiles ===")
print("25th percentile:", np.percentile(data, 25))
print("50th percentile (median):", np.percentile(data, 50))
print("75th percentile:", np.percentile(data, 75))
print("Quartiles:", np.percentile(data, [25, 50, 75]))

# Random number generation
print("\n=== Random Number Generation ===")

# Uniform distribution [0, 1)
uniform_random = np.random.random(5)
print("Uniform random (0-1):", uniform_random)

# Normal distribution (mean=0, std=1)
normal_random = np.random.randn(5)
print("Normal random:", normal_random)

# Normal distribution with custom mean and std
custom_normal = np.random.normal(loc=100, scale=15, size=10)
print("Normal (mean=100, std=15):", custom_normal)

# Random integers
random_ints = np.random.randint(1, 100, size=10)
print("Random integers (1-100):", random_ints)

# Random choice from array
choices = np.random.choice(['red', 'blue', 'green', 'yellow'], size=5)
print("Random choices:", choices)

# 2D Statistics
print("\n=== 2D Array Statistics ===")
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("Matrix:\n", matrix)
print("Overall mean:", np.mean(matrix))
print("Mean by row (axis=1):", np.mean(matrix, axis=1))
print("Mean by column (axis=0):", np.mean(matrix, axis=0))
print("Sum by row:", np.sum(matrix, axis=1))
print("Sum by column:", np.sum(matrix, axis=0))

# Correlation and Covariance
print("\n=== Correlation and Covariance ===")
# Generate correlated data
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5  # y is correlated with x

print("Covariance matrix:\n", np.cov(x, y))
print("Correlation coefficient:", np.corrcoef(x, y)[0, 1])

# Cumulative operations
print("\n=== Cumulative Operations ===")
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Cumulative sum:", np.cumsum(arr))
print("Cumulative product:", np.cumprod(arr))

# Sorting and ordering statistics
print("\n=== Sorting and Order Statistics ===")
unsorted = np.array([5, 2, 8, 1, 9, 3])
print("Original:", unsorted)
print("Sorted:", np.sort(unsorted))
print("Argsort (indices):", np.argsort(unsorted))
print("Largest 3 values:", np.partition(unsorted, -3)[-3:])

# Histogram
print("\n=== Histogram ===")
random_data = np.random.randn(1000)
hist, bin_edges = np.histogram(random_data, bins=10)
print("Histogram counts:", hist)
print("Bin edges:", bin_edges)
