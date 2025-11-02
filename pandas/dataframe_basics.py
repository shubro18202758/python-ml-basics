"""Pandas DataFrame Basics

This module demonstrates fundamental Pandas operations including:
- DataFrame creation and manipulation
- Data selection and filtering
- Handling missing data
- Data aggregation and grouping
- Basic data transformations
"""

import pandas as pd
import numpy as np

# Creating DataFrames
print("=== Creating DataFrames ===")

# From dictionary
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'City': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin'],
    'Salary': [50000, 60000, 75000, 55000, 68000]
}
df = pd.DataFrame(data_dict)
print("DataFrame from dictionary:\n", df)

# From list of lists
data_list = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df_list = pd.DataFrame(data_list, columns=['Name', 'Age'])
print("\nDataFrame from list:\n", df_list)

# Basic DataFrame information
print("\n=== DataFrame Information ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("\nFirst 3 rows:\n", df.head(3))
print("\nLast 2 rows:\n", df.tail(2))
print("\nDataFrame info:")
print(df.info())
print("\nDescriptive statistics:\n", df.describe())

# Selecting data
print("\n=== Data Selection ===")

# Single column
print("Names column:\n", df['Name'])

# Multiple columns
print("\nName and Salary:\n", df[['Name', 'Salary']])

# Row selection by index
print("\nFirst row:\n", df.iloc[0])
print("\nRows 1-3:\n", df.iloc[1:4])

# Row selection by label
print("\nRow at index 2:\n", df.loc[2])

# Specific cell value
print("\nSalary of person at index 1:", df.loc[1, 'Salary'])

# Filtering data
print("\n=== Data Filtering ===")

# Boolean indexing
print("People older than 28:\n", df[df['Age'] > 28])
print("\nPeople from New York or London:\n", 
      df[df['City'].isin(['New York', 'London'])])

# Multiple conditions
print("\nPeople older than 28 AND salary > 60000:\n",
      df[(df['Age'] > 28) & (df['Salary'] > 60000)])

# Adding new columns
print("\n=== Adding Columns ===")
df['Bonus'] = df['Salary'] * 0.1
df['Total_Compensation'] = df['Salary'] + df['Bonus']
print("DataFrame with new columns:\n", df)

# Handling missing data
print("\n=== Handling Missing Data ===")

# Create DataFrame with missing values
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})
print("DataFrame with missing values:\n", df_missing)

# Check for missing values
print("\nMissing values check:\n", df_missing.isnull())
print("\nSum of missing values per column:\n", df_missing.isnull().sum())

# Drop rows with any missing values
print("\nAfter dropping rows with NaN:\n", df_missing.dropna())

# Fill missing values
print("\nFill NaN with 0:\n", df_missing.fillna(0))
print("\nFill NaN with column mean:\n", df_missing.fillna(df_missing.mean()))

# Data aggregation
print("\n=== Data Aggregation ===")

# Create sample data for grouping
df_sales = pd.DataFrame({
    'Product': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'Sales': [100, 150, 200, 120, 180, 160],
    'Quantity': [5, 8, 10, 6, 9, 7]
})
print("Sales data:\n", df_sales)

# Group by and aggregate
print("\nTotal sales by product:\n", df_sales.groupby('Product')['Sales'].sum())
print("\nAverage sales by region:\n", df_sales.groupby('Region')['Sales'].mean())

# Multiple aggregations
print("\nMultiple aggregations by product:\n",
      df_sales.groupby('Product').agg({
          'Sales': ['sum', 'mean', 'count'],
          'Quantity': 'sum'
      }))

# Sorting
print("\n=== Sorting ===")
print("Sort by Age (ascending):\n", df.sort_values('Age'))
print("\nSort by Salary (descending):\n", df.sort_values('Salary', ascending=False))
print("\nSort by multiple columns:\n", 
      df.sort_values(['City', 'Age']))

# Basic operations
print("\n=== Basic Operations ===")
print("Unique cities:", df['City'].unique())
print("Number of unique cities:", df['City'].nunique())
print("Value counts for City:\n", df['City'].value_counts())
print("Average salary:", df['Salary'].mean())
print("Total salary sum:", df['Salary'].sum())
