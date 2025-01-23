import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



data = pd.read_csv('../Files/filtered_output.csv')


X = data[['queue1', 'queue2']]
y = data['action']

# Create a results file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'knn_results_{timestamp}.txt'


with open(results_file, 'w') as f:
    # Write initial information
    f.write("KNN Classification Results\n")
    f.write("="*50 + "\n\n")
    
    # 1. Data Splitting
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Write dataset sizes
    f.write("Dataset Sizes:\n")
    f.write(f"Training set: {len(X_train)} samples\n")
    f.write(f"Validation set: {len(X_val)} samples\n")
    f.write(f"Test set: {len(X_test)} samples\n\n")
    
    # 2. Cross-Validation Setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Grid Search
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    f.write("Grid Search Parameters:\n")
    f.write(str(param_grid) + "\n\n")
    
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
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Write all results from grid search
    f.write("Grid Search Results:\n")
    f.write("-"*50 + "\n")
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Sort results by mean test score
    results = results.sort_values('mean_test_score', ascending=False)
    
    for idx, row in results.iterrows():
        params = row['params']
        f.write(f"\nParameters: {params}\n")
        f.write(f"Mean CV Score: {row['mean_test_score']:.3f} (+/- {row['std_test_score']*2:.3f})\n")
        f.write(f"Mean Train Score: {row['mean_train_score']:.3f}\n")
    
    f.write("\n" + "="*50 + "\n")
    f.write("Best Model Results:\n")
    f.write(f"Best parameters: {grid_search.best_params_}\n")
    f.write(f"Best cross-validation score: {grid_search.best_score_:.3f}\n\n")
    
    # 4. Best Model Validation
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(X_val)
    
    f.write("Validation Set Performance:\n")
    f.write(classification_report(y_val, val_predictions))
    f.write("\n")
    
    # 5. Final Model Evaluation
    test_predictions = best_model.predict(X_test)
    
    f.write("Test Set Performance:\n")
    f.write(classification_report(y_test, test_predictions))
    
    # Write confusion matrix
    f.write("\nConfusion Matrix:\n")
    cm = confusion_matrix(y_test, test_predictions)
    f.write(str(cm))
    
    # Additional metrics
    y_proba = best_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    f.write(f"\n\nROC AUC Score (One-vs-Rest): {roc_auc:.3f}\n")
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_macro')
    f.write("\nFinal Cross-validation Scores:\n")
    f.write(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})\n")

print(f"Results have been saved to: {results_file}")

# Also create visualizations
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