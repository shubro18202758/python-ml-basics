# Linear Algebra for Machine Learning in Fraud Detection

## Core Concepts

### 1. Vectors and Matrices in Transaction Data

#### Transaction as a Vector

Each transaction can be represented as a feature vector:

\[
\mathbf{x} = \begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix} = \begin{bmatrix}
\text{amount} \\ \text{hour} \\ \text{distance} \\ \text{velocity} \\ \vdots
\end{bmatrix}
\]

```python
import numpy as np

# Example transaction vector
transaction = np.array([
    150.50,  # amount
    23,      # hour (11 PM)
    500,     # distance from last txn (km)
    250,     # velocity (km/hr)
    1,       # is_international
    0        # card_present
])

print(f"Transaction vector shape: {transaction.shape}")
print(f"Transaction: {transaction}")
```

#### Dataset as a Matrix

\[
\mathbf{X} = \begin{bmatrix}
\mathbf{x}_1^T \\
\mathbf{x}_2^T \\
\vdots \\
\mathbf{x}_m^T
\end{bmatrix} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1n} \\
x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}
\]

Rows = transactions, Columns = features

---

### 2. Distance Metrics

#### Euclidean Distance

**Most common distance metric:**

\[
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2} = \|\mathbf{x} - \mathbf{y}\|_2
\]

```python
def euclidean_distance(x, y):
    """
    Calculate Euclidean distance between two transaction vectors
    """
    return np.sqrt(np.sum((x - y)**2))

# Example: Distance between current transaction and user's typical transaction
user_avg_transaction = np.array([50.0, 14, 10, 5, 0, 1])
current_transaction = np.array([500.0, 23, 200, 150, 1, 0])

distance = euclidean_distance(current_transaction, user_avg_transaction)
print(f"Distance from typical behavior: {distance:.2f}")
```

#### Manhattan Distance

**L1 norm, less sensitive to outliers:**

\[
d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} |x_i - y_i| = \|\mathbf{x} - \mathbf{y}\|_1
\]

```python
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))
```

#### Cosine Similarity

**Measures angle between vectors (useful for high-dimensional sparse data):**

\[
\text{similarity}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_{i=1}^{n} x_i y_i}{\sqrt{\sum_{i=1}^{n} x_i^2} \sqrt{\sum_{i=1}^{n} y_i^2}}
\]

```python
def cosine_similarity(x, y):
    """
    Cosine similarity (ranges from -1 to 1)
    """
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    return dot_product / (norm_x * norm_y + 1e-10)

def cosine_distance(x, y):
    """
    Cosine distance (1 - similarity)
    """
    return 1 - cosine_similarity(x, y)
```

---

### 3. Matrix Operations in ML

#### Linear Transformation

**Transforming transaction features:**

\[
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}
\]

```python
def linear_transformation(X, W, b):
    """
    Apply linear transformation to features
    
    Args:
        X: (m, n) feature matrix
        W: (k, n) weight matrix
        b: (k,) bias vector
    
    Returns:
        Y: (m, k) transformed features
    """
    return np.dot(X, W.T) + b

# Example: 6 features -> 3 hidden features
X = np.random.randn(100, 6)  # 100 transactions, 6 features
W = np.random.randn(3, 6)     # 3 hidden units
b = np.random.randn(3)

Y = linear_transformation(X, W, b)
print(f"Original shape: {X.shape}")
print(f"Transformed shape: {Y.shape}")
```

#### Matrix Multiplication in Neural Networks

```python
class FraudDetectorLayer:
    def __init__(self, input_dim, output_dim):
        # Xavier initialization
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(output_dim)
    
    def forward(self, X):
        """
        Forward pass: Y = activation(WX + b)
        """
        self.X = X
        self.Z = np.dot(X, self.W.T) + self.b
        self.A = self.relu(self.Z)
        return self.A
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def backward(self, dA, learning_rate=0.01):
        """
        Backward pass for gradient descent
        """
        m = self.X.shape[0]
        
        # ReLU gradient
        dZ = dA * (self.Z > 0)
        
        # Gradients
        dW = (1/m) * np.dot(dZ.T, self.X)
        db = (1/m) * np.sum(dZ, axis=0)
        dX = np.dot(dZ, self.W)
        
        # Update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX
```

---

### 4. Eigenvalues and Eigenvectors

#### Understanding Eigendecomposition

**For a square matrix A:**

\[
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
\]

Where:
- \(\mathbf{v}\): Eigenvector (direction)
- \(\lambda\): Eigenvalue (scaling factor)

#### Principal Component Analysis (PCA)

**PCA finds directions of maximum variance:**

```python
import numpy as np
from numpy.linalg import eig

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        """
        Fit PCA using eigendecomposition of covariance matrix
        """
        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Covariance matrix
        cov = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eig(cov)
        
        # Sort by eigenvalues (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store top k eigenvectors
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigenvalues)
        
        return self
    
    def transform(self, X):
        """
        Project data onto principal components
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X_transformed):
        """
        Reconstruct original features
        """
        return np.dot(X_transformed, self.components.T) + self.mean

# Usage for fraud detection
pca = PCA(n_components=5)
pca.fit(X_train)

print(f"Explained variance ratio: {pca.explained_variance_ratio}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio):.2%}")

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

---

### 5. Singular Value Decomposition (SVD)

**Factorize any matrix:**

\[
\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
\]

Where:
- \(\mathbf{U}\): Left singular vectors (m × m)
- \(\mathbf{\Sigma}\): Diagonal matrix of singular values (m × n)
- \(\mathbf{V}^T\): Right singular vectors (n × n)

```python
def svd_dimensionality_reduction(X, n_components):
    """
    Reduce dimensionality using SVD (more numerically stable than PCA)
    """
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Keep top k components
    U_reduced = U[:, :n_components]
    S_reduced = S[:n_components]
    Vt_reduced = Vt[:n_components, :]
    
    # Transform
    X_transformed = U_reduced * S_reduced
    
    # Reconstruction
    X_reconstructed = np.dot(U_reduced * S_reduced, Vt_reduced)
    
    return X_transformed, X_reconstructed

# Usage
X_reduced, X_reconstructed = svd_dimensionality_reduction(X, n_components=10)

# Reconstruction error (fraud indicator)
reconstruction_error = np.sum((X - X_reconstructed)**2, axis=1)
anomalies = reconstruction_error > np.percentile(reconstruction_error, 99)
```

---

### 6. Matrix Norms

#### Frobenius Norm

**Measure matrix magnitude:**

\[
\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}^2}
\]

```python
def frobenius_norm(A):
    return np.sqrt(np.sum(A**2))

# Or use numpy
frobenius = np.linalg.norm(A, 'fro')
```

#### Regularization in Fraud Models

**L2 regularization (Ridge):**

\[
\min_{\mathbf{w}} \|\mathbf{Xw} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2
\]

**L1 regularization (Lasso):**

\[
\min_{\mathbf{w}} \|\mathbf{Xw} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_1
\]

```python
from sklearn.linear_model import Ridge, Lasso

# L2 regularization (prevents overfitting)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# L1 regularization (feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print(f"Non-zero coefficients (Lasso): {np.sum(lasso.coef_ != 0)}")
```

---

### 7. Matrix Inverse and Pseudo-Inverse

#### Solving Linear Systems

**Normal equation for linear regression:**

\[
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
\]

```python
def linear_regression_normal_equation(X, y):
    """
    Solve linear regression using matrix inversion
    """
    # Add bias term
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Normal equation
    XtX = np.dot(X_bias.T, X_bias)
    Xty = np.dot(X_bias.T, y)
    
    # Solve (with regularization for stability)
    w = np.linalg.solve(XtX + 1e-6 * np.eye(XtX.shape[0]), Xty)
    
    return w
```

#### Moore-Penrose Pseudo-Inverse

**For non-square or singular matrices:**

\[
\mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T
\]

```python
def pseudo_inverse_regression(X, y):
    """
    Linear regression using pseudo-inverse (more stable)
    """
    X_bias = np.c_[np.ones(X.shape[0]), X]
    X_pinv = np.linalg.pinv(X_bias)
    w = np.dot(X_pinv, y)
    return w
```

---

### 8. Quadratic Forms

**Used in Mahalanobis distance and elliptical decision boundaries:**

\[
Q(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}
\]

```python
def quadratic_form_anomaly(x, A):
    """
    Calculate quadratic form (used in Mahalanobis distance)
    
    Args:
        x: Transaction feature vector
        A: Positive definite matrix (often inverse covariance)
    """
    return np.dot(np.dot(x.T, A), x)

# Example: Gaussian discriminant analysis decision boundary
def gda_decision_boundary(x, mu_0, mu_1, Sigma_inv, prior_0, prior_1):
    """
    Quadratic decision boundary for Gaussian Discriminant Analysis
    """
    diff_0 = x - mu_0
    diff_1 = x - mu_1
    
    score_0 = -0.5 * quadratic_form_anomaly(diff_0, Sigma_inv) + np.log(prior_0)
    score_1 = -0.5 * quadratic_form_anomaly(diff_1, Sigma_inv) + np.log(prior_1)
    
    return 1 if score_1 > score_0 else 0
```

---

### 9. Gradient Computation

**Gradient of loss function with respect to weights:**

\[
\nabla_{\mathbf{w}} L = \frac{\partial L}{\partial \mathbf{w}} = \mathbf{X}^T (\mathbf{Xw} - \mathbf{y})
\]

```python
def compute_gradient(X, y, w):
    """
    Compute gradient for logistic regression
    """
    m = X.shape[0]
    
    # Predictions
    z = np.dot(X, w)
    predictions = 1 / (1 + np.exp(-z))  # Sigmoid
    
    # Gradient
    gradient = (1/m) * np.dot(X.T, (predictions - y))
    
    return gradient

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Train logistic regression using gradient descent
    """
    m, n = X.shape
    w = np.zeros(n)
    
    for i in range(n_iterations):
        gradient = compute_gradient(X, y, w)
        w -= learning_rate * gradient
        
        if i % 100 == 0:
            loss = binary_cross_entropy(X, y, w)
            print(f"Iteration {i}, Loss: {loss:.4f}")
    
    return w

def binary_cross_entropy(X, y, w):
    """
    Calculate binary cross-entropy loss
    """
    m = X.shape[0]
    z = np.dot(X, w)
    predictions = 1 / (1 + np.exp(-z))
    
    loss = -(1/m) * np.sum(y * np.log(predictions + 1e-10) + 
                           (1 - y) * np.log(1 - predictions + 1e-10))
    return loss
```

---

## Key Takeaways

1. **Vectors & Matrices**: Foundation for representing transaction data
2. **Distance Metrics**: Essential for anomaly detection algorithms
3. **Eigendecomposition**: Powers PCA for dimensionality reduction
4. **SVD**: Robust alternative for large-scale fraud detection
5. **Gradient Descent**: Core optimization algorithm for training models

---

## Practice Problems

1. Implement k-NN fraud detector using different distance metrics
2. Build PCA from scratch and compare with scikit-learn
3. Create anomaly detector using reconstruction error from SVD
4. Implement batch gradient descent for logistic regression

---

## References

1. Strang, G. (2016). *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
2. Trefethen, L. N., & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
