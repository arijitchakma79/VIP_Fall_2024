import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read data
print("Loading dataset...")
data = pd.read_csv('./filtered_output.csv')
print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns.")

X = data[['queue1', 'queue2']]
y = data['action']

# Create results file
results_file = "knn_results.txt"
print(f"Results will be saved to: {results_file}")

with open(results_file, 'w') as f:
    def write_line(text=""):
        """Write a line to the TXT file and flush immediately."""
        f.write(text + "\n")
        f.flush()  # Ensures writing is done line by line

    # Writing the title
    write_line("KNN Classification Results")
    write_line("=" * 50)
    write_line()

    # 1. Data Splitting
    print("Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    print(f"Data split completed: Training={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    write_line("Dataset Sizes:")
    write_line(f"Training set: {len(X_train)} samples")
    write_line(f"Validation set: {len(X_val)} samples")
    write_line(f"Test set: {len(X_test)} samples")
    write_line()

    # 2. Cross-Validation Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Grid Search
    print("Initializing Grid Search...")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    write_line("Grid Search Parameters:")
    for key, values in param_grid.items():
        write_line(f"{key}: {values}")
    write_line()

    # Fit Grid Search
    print("Fitting Grid Search...")
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn,
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    print("Grid Search completed!")

    # Best Model Selection
    best_model = grid_search.best_estimator_
    write_line("Best Model Results:")
    write_line(f"Best parameters: {grid_search.best_params_}")
    write_line(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    write_line()

    # 4. Best Model Validation
    print("Validating model on validation set...")
    val_predictions = best_model.predict(X_val)
    print("Validation completed.")

    write_line("Validation Set Performance:")
    write_line(classification_report(y_val, val_predictions))
    write_line()

    # 5. Final Model Evaluation
    print("Evaluating model on test set...")
    test_predictions = best_model.predict(X_test)
    print("Test evaluation completed.")

    write_line("Test Set Performance:")
    write_line(classification_report(y_test, test_predictions))
    write_line()

    # Confusion Matrix
    print("Computing confusion matrix...")
    cm = confusion_matrix(y_test, test_predictions)
    write_line("Confusion Matrix:")
    write_line(str(cm))

    # Additional metrics
    y_proba = best_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    write_line(f"\nROC AUC Score (One-vs-Rest): {roc_auc:.3f}")

    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_macro')
    write_line("\nFinal Cross-validation Scores:")
    write_line(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})\n")

print(f"TXT Report saved: {results_file}")

# Also create visualizations
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
plt.title('KNN Decision Boundaries on Test Set')
plt.xlabel('Queue 1')
plt.ylabel('Queue 2')

plt.tight_layout()
plt.show()
print("Visualization completed.")

print("All processes completed successfully!")
