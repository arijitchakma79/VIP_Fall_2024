import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read data
print("Loading dataset...")
data = pd.read_csv('paste.txt')
print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns.")

# Prepare features and target
X = data[['queue1', 'queue2']]
y = data['action']
print("Features and target selected.")

# Create results file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'random_forest_results_{timestamp}.txt'
print(f"Results will be saved to: {results_file}")

with open(results_file, 'w') as f:
    f.write("Random Forest Classification Results\n")
    f.write("="*50 + "\n\n")

    # Data Splitting
    print("Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(f"Data split completed: Training={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    # Grid Search Parameters
    print("Initializing Grid Search...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    print("Grid Search parameters defined.")

    # Grid Search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("Fitting Grid Search (this may take time)...")
    grid_search.fit(X_train, y_train)
    print("Grid Search completed!")

    # Best Model Selection
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Feature Importance
    print("Extracting feature importance...")
    feature_importances = best_model.feature_importances_

    # Model Validation
    print("Validating model on validation set...")
    val_predictions = best_model.predict(X_val)
    print("Validation completed.")

    # Model Evaluation on Test Set
    print("Evaluating model on test set...")
    test_predictions = best_model.predict(X_test)
    print("Test evaluation completed.")

    # Confusion Matrix
    print("Computing confusion matrix...")
    cm = confusion_matrix(y_test, test_predictions)

    # Additional Metrics
    print("Computing ROC AUC score...")
    y_proba = best_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

print(f"Results have been saved to: {results_file}")

# Create visualizations
print("Generating visualizations...")

plt.figure(figsize=(20, 5))

# Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Feature Importance
plt.subplot(1, 3, 2)
importances = pd.DataFrame({
    'feature': ['Queue1', 'Queue2'],
    'importance': best_model.feature_importances_
})
sns.barplot(x='feature', y='importance', data=importances)
plt.title('Feature Importance')

# Decision Boundaries
plt.subplot(1, 3, 3)
h = 0.02
x_min, x_max = X['queue1'].min() - 1, X['queue1'].max() + 1
y_min, y_max = X['queue2'].min() - 1, X['queue2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_test['queue1'], X_test['queue2'], c=y_test, alpha=0.8)
plt.title('Random Forest Decision Boundaries')
plt.xlabel('Queue 1')
plt.ylabel('Queue 2')

plt.tight_layout()
plt.show()
print("Visualization completed.")

print("All processes completed successfully!")
