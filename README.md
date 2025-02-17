

# 📚 Machine Learning Cheat Sheet

### Table of Contents
1. [Terminology](#terminology)
2. [Supervised Learning](#supervised-learning)
   - [Regression](#regression)
   - [Classification](#classification)
3. [Unsupervised Learning](#unsupervised-learning)
   - [Clustering](#clustering)
   - [Dimensionality Reduction](#dimensionality-reduction)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Common Algorithms](#common-algorithms)
6. [Feature Engineering](#feature-engineering)
7. [Model Optimization](#model-optimization)
8. [Cross-Validation](#cross-validation)
9. [Overfitting & Underfitting](#overfitting--underfitting)
10. [Useful Python Libraries](#useful-python-libraries)

---

## Terminology
- **Feature**: Independent variable, input to the model.
- **Label/Target**: Dependent variable, output we predict.
- **Model**: Mathematical representation of the data.
- **Training Set**: Subset of data used to fit the model.
- **Test Set**: Subset of data used to evaluate the model.

---

## Supervised Learning
Supervised learning uses labeled data to predict outcomes.

### Regression
**Goal**: Predict continuous values (e.g., house prices).

- **Algorithms**:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Decision Trees
  - Random Forests
  - Support Vector Regression (SVR)

**Common Loss Functions**:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

### Classification
**Goal**: Predict discrete values (e.g., spam/not spam).

- **Algorithms**:
  - Logistic Regression
  - k-Nearest Neighbors (k-NN)
  - Support Vector Machines (SVM)
  - Naive Bayes
  - Decision Trees
  - Random Forest
  - Neural Networks

**Evaluation Metrics**:
- Accuracy
- Precision, Recall
- F1-Score
- ROC-AUC Score

---

## Unsupervised Learning
Unsupervised learning works on unlabeled data to find patterns.

### Clustering
**Goal**: Group similar data points together.

- **Algorithms**:
  - k-Means
  - DBSCAN
  - Hierarchical Clustering
  - Gaussian Mixture Models (GMM)

### Dimensionality Reduction
**Goal**: Reduce the number of features while preserving important information.

- **Algorithms**:
  - Principal Component Analysis (PCA)
  - t-Distributed Stochastic Neighbor Embedding (t-SNE)
  - Linear Discriminant Analysis (LDA)

---

## Evaluation Metrics
### Classification Metrics
- **Accuracy**: \( \frac{TP + TN}{TP + TN + FP + FN} \)
- **Precision**: \( \frac{TP}{TP + FP} \)
- **Recall (Sensitivity)**: \( \frac{TP}{TP + FN} \)
- **F1-Score**: \( 2 \times \frac{Precision \times Recall}{Precision + Recall} \)
- **ROC Curve**: Plots true positive rate (TPR) against false positive rate (FPR).
- **AUC**: Area under the ROC curve, higher means better.

### Regression Metrics
- **Mean Squared Error (MSE)**: \( \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y_i})^2 \)
- **Mean Absolute Error (MAE)**: \( \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}| \)
- **R² Score**: \( 1 - \frac{SS_{res}}{SS_{tot}} \)

---

## Common Algorithms
| Algorithm               | Type           | Use Case                   |
|-------------------------|----------------|----------------------------|
| Linear Regression        | Regression     | Predict continuous values   |
| Logistic Regression      | Classification | Binary/multiclass problems  |
| Decision Tree            | Both           | Simple models, interpretable|
| Random Forest            | Both           | Robust, ensemble technique  |
| k-NN                     | Both           | Simple, interpretable       |
| SVM                      | Both           | High-dimensional spaces     |
| k-Means                  | Clustering     | Unsupervised clustering     |
| PCA                      | Dim. Reduction | Reduce feature space        |
| Naive Bayes              | Classification | Text classification         |
| Neural Networks (NN)     | Both           | Complex problems            |

---

## Feature Engineering
1. **Normalisation/Standardisation**:
   - **Normalisation**: Rescale data to [0, 1].
   - **Standardisation**: Rescale data to have mean 0, variance 1.

2. **Handling Missing Data**:
   - Remove missing values.
   - Impute missing values using the mean, median, or mode.

3. **Encoding Categorical Data**:
   - Label Encoding
   - One-Hot Encoding

4. **Feature Creation**:
   - Polynomial Features
   - Log Transform
   - Interaction Terms

---

## Model Optimisation
1. **Hyperparameter Tuning**:
   - Use Grid Search or Random Search to find the best parameters.
   - Libraries: `GridSearchCV`, `RandomizedSearchCV`.

2. **Regularisation**:
   - **L1 Regularisation (Lasso)**: Shrinks coefficients to zero, useful for feature selection.
   - **L2 Regularisation (Ridge)**: Penalizes large coefficients to reduce overfitting.

---

## Cross-Validation
- **k-Fold Cross-Validation**:
  - Split data into `k` subsets, train on `k-1` subsets, test on the last subset, repeat.
  - Average performance over all `k` trials.
  
- **Stratified k-Fold**:
  - Ensures each fold has the same proportion of classes as the original dataset (useful for classification).

---

## Overfitting & Underfitting
- **Overfitting**: Model performs well on training data but poorly on test data.
  - **Solution**: Use regularisation, collect more data, reduce model complexity.
  
- **Underfitting**: Model performs poorly on both training and test data.
  - **Solution**: Use a more complex model, add more features.

---

## Useful Python Libraries
- **NumPy**: `pip install numpy`
- **Pandas**: `pip install pandas`
- **Scikit-learn**: `pip install scikit-learn`
- **TensorFlow**: `pip install tensorflow`
- **Keras**: `pip install keras`
- **PyTorch**: `pip install torch`
- **XGBoost**: `pip install xgboost`
- **LightGBM**: `pip install lightgbm`
- **Matplotlib**: `pip install matplotlib`
- **Seaborn**: `pip install seaborn`

---



