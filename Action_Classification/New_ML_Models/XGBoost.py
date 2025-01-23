import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read data
data = pd.read_csv('paste.txt')
X = data[['queue1', 'queue2']]
y = data['action']

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f'xgboost_results_{timestamp}.txt'

with open(results_file, 'w') as f:
    f.write("XGBoost Classification Results\n")
    f.write("="*50 + "\n\n")
    
    # Data Splitting
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    # Grid Search Parameters
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    f.write("Dataset Sizes:\n")
    f.write(f"Training set: {len(X_train)} samples\n")
    f.write(f"Validation set: {len(X_val)} samples\n")
    f.write(f"Test set: {len(X_test)} samples\n\n")
    
    f.write("Grid Search Parameters:\n")
    f.write(str(param_grid) + "\n\n")
    
    # Grid Search
    xgb = XGBClassifier(objective='multi:softprob', random_state=42)
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values('mean_test_score', ascending=False)
    
    f.write("Top 5 Parameter Combinations:\n")
    for idx, row in results.head().iterrows():
        f.write(f"\nParameters: {row['params']}\n")
        f.write(f"Mean CV Score: {row['mean_test_score']:.3f} (+/- {row['std_test_score']*2:.3f})\n")
    
    best_model = grid_search.best_estimator_
    
    f.write("\nBest Parameters:\n")
    f.write(str(grid_search.best_params_))
    f.write(f"\nBest CV Score: {grid_search.best_score_:.3f}\n")
    
    # Feature Importance
    f.write("\nFeature Importance:\n")
    feature_importance = best_model.feature_importances_
    for feat, imp in zip(['queue1', 'queue2'], feature_importance):
        f.write(f"{feat}: {imp:.4f}\n")
    
    # Model Evaluation
    val_predictions = best_model.predict(X_val)
    test_predictions = best_model.predict(X_test)
    
    f.write("\nValidation Set Performance:\n")
    f.write(classification_report(y_val, val_predictions))
    
    f.write("\nTest Set Performance:\n")
    f.write(classification_report(y_test, test_predictions))
    
    cm = confusion_matrix(y_test, test_predictions)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))
    
    
# Visualizations
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
    'importance': feature_importance
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
plt.title('XGBoost Decision Boundaries')
plt.xlabel('Queue 1')
plt.ylabel('Queue 2')

plt.tight_layout()
plt.show()

# Learning curves
train_scores = []
val_scores = []
estimators = range(10, 310, 10)

for n_estimators in estimators:
    model = XGBClassifier(**grid_search.best_params_, n_estimators=n_estimators)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    val_scores.append(model.score(X_val, y_val))

plt.figure(figsize=(10, 5))
plt.plot(estimators, train_scores, label='Training Score')
plt.plot(estimators, val_scores, label='Validation Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.show()