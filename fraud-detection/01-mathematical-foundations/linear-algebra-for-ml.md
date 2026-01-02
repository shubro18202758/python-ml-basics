# Linear Algebra for Machine Learning in Fraud Detection

## Core Concepts

### 1. Vectors and Matrices in Transaction Data

**Vector Representation of Transactions**

In machine learning and fraud detection, each transaction is represented as a feature vector. This vector representation is fundamental because it allows us to apply powerful linear algebra operations and geometric interpretations to transaction data.

A transaction vector can be represented as:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} \text{amount} \\ \text{hour} \\ \text{distance} \\ \text{velocity} \end{bmatrix}$$

Each element represents a transaction feature:
- $x_1$: Transaction amount (currency units)
- $x_2$: Hour of transaction (0-23)
- $x_3$: Distance from last transaction location (km)
- $x_4$: Velocity between locations (km/hr)
- Additional features capturing merchant category, device info, IP location, etc.

**Why Vector Representation Matters:**

Vectorization enables:
- **Efficient Computation**: Matrix operations can be parallelized on GPUs
- **Geometric Intuition**: Distance metrics measure transaction similarity
- **Standardized Processing**: All transactions handled uniformly
- **Scalability**: Millions of transactions processed simultaneously

### 2. Matrices and Transaction Datasets

**Dataset as a Matrix**

A dataset of $m$ transactions with $n$ features forms an $m \times n$ matrix:

$$\mathbf{X} = \begin{bmatrix} x_{1,1} & x_{1,2} & \cdots & x_{1,n} \\ x_{2,1} & x_{2,2} & \cdots & x_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ x_{m,1} & x_{m,2} & \cdots & x_{m,n} \end{bmatrix}$$

Where $x_{i,j}$ is the $j$-th feature of the $i$-th transaction.

**Practical Example:**

A dataset with 1,000 transactions and 8 features creates a $1000 \times 8$ matrix. Each row is a transaction vector, and each column represents a feature across all transactions.

### 3. Distance Metrics for Anomaly Detection

**Euclidean Distance**

The Euclidean distance between two transactions measures how different they are:

$$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{k=1}^{n} (x_{i,k} - x_{j,k})^2}$$

**Application**: Find nearest neighbors. Transactions far from the normal behavior cluster are likely fraudulent.

**Manhattan Distance (L1)**

Alternative distance metric that's faster to compute:

$$d(\mathbf{x}_i, \mathbf{x}_j) = \sum_{k=1}^{n} |x_{i,k} - x_{j,k}|$$

**Application**: Feature-wise deviation analysis. Identifies which specific features deviate most from normal.

**Mahalanobis Distance**

Accounts for feature correlations and scaling:

$$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{(\mathbf{x}_i - \mathbf{x}_j)^T \mathbf{\Sigma}^{-1} (\mathbf{x}_i - \mathbf{x}_j)}$$

Where $\mathbf{\Sigma}$ is the covariance matrix of the feature distribution.

**Advantage**: Detects anomalies considering both magnitude and correlations. A transaction might be individually normal but unusual when considering feature relationships.

### 4. Eigenvalues and Eigenvectors

**Understanding Eigenvectors**

For a square matrix $\mathbf{A}$, an eigenvector $\mathbf{v}$ satisfies:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

Where $\lambda$ is the eigenvalue.

**Application in PCA (Principal Component Analysis)**

Eigenvectors of the covariance matrix identify directions of maximum variance in transaction data:

$$\mathbf{C} = \frac{1}{m} \mathbf{X}^T \mathbf{X}$$

Eigenvectors point toward directions where transactions vary most. Transactions projecting far along unusual eigenvectors are anomalous.

### 5. Singular Value Decomposition (SVD)

**SVD Decomposition**

Any matrix can be decomposed as:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Where:
- $\mathbf{U}$: Left singular vectors (transaction space)
- $\mathbf{\Sigma}$: Diagonal matrix of singular values (importance weights)
- $\mathbf{V}^T$: Right singular vectors (feature space)

**Application**: Anomaly detection through reconstruction error. Reconstruct transaction using top-k singular vectors:

$$\mathbf{X}_{\text{reconstructed}} = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T$$

Transactions with large reconstruction error deviate from normal patterns:

$$\text{Anomaly Score} = ||\mathbf{x} - \mathbf{x}_{\text{reconstructed}}||^2$$

**Why SVD Works for Fraud**: Fraudsters create different patterns. Normal transactions lie in a lower-dimensional subspace; fraud falls outside it.

### 6. Covariance Matrix and Feature Relationships

**Covariance Matrix**

Captures correlations between all feature pairs:

$$\mathbf{\Sigma} = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{x}_i - \bar{\mathbf{x}}) (\mathbf{x}_i - \bar{\mathbf{x}})^T$$

Where $\bar{\mathbf{x}}$ is the mean transaction vector.

**Interpretation:**
- $\Sigma_{jj}$ = Variance of feature $j$
- $\Sigma_{jk}$ = Covariance between features $j$ and $k$
- Large positive covariance: Features move together
- Negative covariance: Inverse relationship

**Fraud Detection Application**: Fraudsters often break normal feature correlations. For example:
- Legitimate: High amount + evening hour (strong positive correlation)
- Fraudulent: High amount + 3 AM (unexpected combination)

## Matrix Operations in Practice

**Feature Scaling**

Normalizing features to unit variance:

$$\mathbf{X}_{\text{scaled}} = \mathbf{X} \cdot \text{diag}(\mathbf{\sigma})^{-1}$$

Where $\mathbf{\sigma}$ is the standard deviation vector.

**Why Scaling Matters**: Prevents high-magnitude features (e.g., amount in millions) from dominating distance calculations.

**Principal Component Analysis Formula**

Project high-dimensional data to lower dimensions:

$$\mathbf{X}_{\text{PCA}} = \mathbf{X} \mathbf{W}$$

Where $\mathbf{W}$ contains the top-k eigenvectors as columns.

**Computational Efficiency**: Reduces features from 50+ to 10-15 principal components while preserving 95% of variance.

## References

- Golub, G. H., & Van Loan, C. F. (2013). Matrix computations. Johns Hopkins University Press.
- Strang, G. (2009). Introduction to linear algebra. Wellesley-Cambridge Press.
- Jolliffe, I. T. (2002). Principal component analysis. Springer Science+Business Media.
- Mardia, K. V., Kent, J. T., & Bibby, J. M. (1979). Multivariate analysis. Academic press.
