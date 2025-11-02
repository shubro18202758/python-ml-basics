"""Matplotlib Plotting Examples

This module demonstrates fundamental Matplotlib plotting techniques including:
- Line plots and scatter plots
- Bar charts and histograms
- Subplots and figure customization
- Plot styling and annotations
"""

import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Example 1: Basic Line Plot
print("=== Creating Line Plot ===")
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, np.cos(x), label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('line_plot.png', dpi=300, bbox_inches='tight')
print("Line plot saved as 'line_plot.png'")
plt.close()

# Example 2: Scatter Plot
print("\n=== Creating Scatter Plot ===")
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5

plt.figure(figsize=(8, 6))
plt.scatter(x_scatter, y_scatter, c='purple', alpha=0.6, s=50)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Random Data')
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved as 'scatter_plot.png'")
plt.close()

# Example 3: Bar Chart
print("\n=== Creating Bar Chart ===")
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
print("Bar chart saved as 'bar_chart.png'")
plt.close()

# Example 4: Histogram
print("\n=== Creating Histogram ===")
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normal Distribution')
plt.grid(True, alpha=0.3)
plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
print("Histogram saved as 'histogram.png'")
plt.close()

# Example 5: Multiple Subplots
print("\n=== Creating Subplots ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x), 'r-')
axes[0, 0].set_title('Sine Wave')
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Scatter plot
x_scatter = np.random.randn(50)
y_scatter = np.random.randn(50)
axes[0, 1].scatter(x_scatter, y_scatter, c='green', alpha=0.6)
axes[0, 1].set_title('Scatter Plot')
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Bar chart
categories = ['A', 'B', 'C', 'D']
values = [10, 24, 36, 18]
axes[1, 0].bar(categories, values, color='orange')
axes[1, 0].set_title('Bar Chart')

# Subplot 4: Histogram
data = np.random.randn(500)
axes[1, 1].hist(data, bins=20, color='purple', alpha=0.7)
axes[1, 1].set_title('Histogram')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('subplots.png', dpi=300, bbox_inches='tight')
print("Subplots saved as 'subplots.png'")
plt.close()

# Example 6: Customized Plot with Annotations
print("\n=== Creating Customized Plot with Annotations ===")
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2.5)

# Add annotations
max_idx = np.argmax(y)
plt.plot(x[max_idx], y[max_idx], 'ro', markersize=10)
plt.annotate('Maximum',
             xy=(x[max_idx], y[max_idx]),
             xytext=(x[max_idx]+0.5, y[max_idx]-0.3),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=12,
             color='red')

plt.xlabel('X-axis (radians)', fontsize=12)
plt.ylabel('Y-axis', fontsize=12)
plt.title('Sine Wave with Annotation', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.savefig('annotated_plot.png', dpi=300, bbox_inches='tight')
print("Annotated plot saved as 'annotated_plot.png'")
plt.close()

# Example 7: Pie Chart
print("\n=== Creating Pie Chart ===")
sizes = [35, 25, 20, 15, 5]
labels = ['A', 'B', 'C', 'D', 'E']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']
explode = (0.1, 0, 0, 0, 0)  # Explode first slice

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Pie Chart Example')
plt.axis('equal')  # Equal aspect ratio ensures circular pie
plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
print("Pie chart saved as 'pie_chart.png'")
plt.close()

print("\nAll plots created successfully!")
print("Note: To display plots interactively, use plt.show() instead of plt.savefig()")
