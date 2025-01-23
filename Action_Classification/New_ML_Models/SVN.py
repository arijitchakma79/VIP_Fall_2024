import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the data
print("Loading dataset...")
data = pd.read_csv('../Files/combined.csv')
print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns.")

# Prepare features and target
X = data[['queue1', 'queue2']]
y = data['action']
print("Features and target selected.")

# Create a results file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'svm_results_{timestamp}.txt'
print(f"Results will be saved to: {results_file}")

with open(results_file, 'w') as f:
    # Write initial information
    f.write("SVM Classification Results\n")
    f.write("="*50 + "\n\n")
    
    # 1. Data Splitting
    print("Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(f"Data split completed: Training={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    # 2. Cross-Validation Setup
    print("Setting up cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Grid Search
    print("Initializing Grid Search...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'degree': [2, 3] # for poly kernel
    }
    
    print("Grid Search parameters defined.")
    
    svm = SVC(probability=True)
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("Fitting Grid Search...")
    grid_search.fit(X_train, y_train)
    print("Grid Search completed!")

    # 4. Best Model Selection
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # 5. Model Validation
    print("Validating model...")
    val_predictions = best_model.predict(X_val)
    print("Validation completed.")

    # 6. Final Model Evaluation
    print("Evaluating on test set...")
    test_predictions = best_model.predict(X_test)
    print("Test evaluation completed.")

    # Confusion Matrix
    print("Computing confusion matrix...")
    cm = confusion_matrix(y_test, test_predictions)

    # Cross-validation scores
    print("Performing cross-validation on final model...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"Cross-validation completed. Mean score: {cv_scores.mean():.3f}")

print(f"Results have been saved to: {results_file}")

# Create visualizations
print("Generating visualizations...")

plt.figure(figsize=(15, 5))

# Confusion Matrix
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Decision Boundaries
plt.subplot(1, 2, 2)
h = 0.02
x_min, x_max = X['queue1'].min() - 1, X['queue1'].max() + 1
y_min, y_max = X['queue2'].min() - 1, X['queue2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_test['queue1'], X_test['queue2'], c=y_test, alpha=0.8)
plt.title('SVM Decision Boundaries on Test Set')
plt.xlabel('Queue 1')
plt.ylabel('Queue 2')

plt.tight_layout()
plt.show()

print("Visualization completed.")

# Additional plot for kernel visualization if using RBF kernel
if best_model.kernel == 'rbf':
    print("Generating support vector coefficients plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(best_model.dual_coef_[0])), 
                np.abs(best_model.dual_coef_[0]),
                alpha=0.5)
    plt.title('Magnitude of Support Vector Coefficients')
    plt.xlabel('Support Vector Index')
    plt.ylabel('Coefficient Magnitude')
    plt.show()
    print("Kernel visualization completed.")

print("All processes completed successfully!")
