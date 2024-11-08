import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from Action_Classifier import Action_Classifier


class RandomForest(Action_Classifier):
    def select_best_parameters(self):
        """
        Use GridSearchCV to find the best hyperparameters for the Random Forest model.
        """
        try:
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Define the parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300, 400,500, 600, 700, 800],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            # Perform grid search with cross-validation
            grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)

            # Store the best model and parameters
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters found: {self.best_params}")
        except Exception as e:
            print("Error during parameter selection:", e)

    def train_model(self):
        """
        Train the Random Forest model using the best parameters found.
        """
        try:
            if not self.best_model:
                raise ValueError("No best model found. Run select_best_parameters() first.")
            
            # Fit the best model to the training data
            self.best_model.fit(self.X_train, self.y_train)
            
            # Make predictions on the test set
            self.y_pred = self.best_model.predict(self.X_test)

            # Print evaluation metrics
            print("\nClassification Report on Test Data:")
            print(classification_report(self.y_test, self.y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(self.y_test, self.y_pred))
        except Exception as e:
            print("Error during model training:", e)

    
    def predict(self, new_data):
        """
        Predicts the class for new data using the trained Random Forest model.
        """
        try:
            if self.best_model is None:
                raise ValueError("Model has not been trained yet. Please train the model before making predictions.")
            
            predictions = self.best_model.predict(new_data)
            return predictions
        except Exception as e:
            print("Error during prediction:", e)
            return None

"""
Data loaded successfully.
Data preprocessing completed.
Best parameters found: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 100}

Classification Report on Test Data:
              precision    recall  f1-score   support

           1       0.70      1.00      0.82         7
           2       0.68      0.70      0.69        40
           4       0.45      0.48      0.46        27
           5       0.14      0.14      0.14         7
           6       1.00      0.33      0.50         6
           7       1.00      0.33      0.50         3

    accuracy                           0.58        90
   macro avg       0.66      0.50      0.52        90
weighted avg       0.60      0.58      0.57        90


Confusion Matrix:
[[ 7  0  0  0  0  0]
 [ 3 28  9  0  0  0]
 [ 0 10 13  4  0  0]
 [ 0  3  3  1  0  0]
 [ 0  0  3  1  2  0]
 [ 0  0  1  1  0  1]]
Predictions for new data: [2 1 2]
/usr/local/lib/python3.10/dist-packages/sklearn/base.py:493: UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
  warnings.warn(
  """