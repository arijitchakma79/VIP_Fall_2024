import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from Action_Classifier import Action_Classifier


class XGBoostClassifier(Action_Classifier):
    def pre_process_data(self):
        """
        Preprocess the data by handling missing values and extracting features and target.
        """
        try:
            X = self.data[self.feature_columns].copy()  
            y = self.data[self.target_column]

            unique_classes = sorted(np.unique(y))
            class_mapping = {label: index for index, label in enumerate(unique_classes)}
            y_mapped = y.map(class_mapping)

            print("Class mapping:", class_mapping)

            self.X = X.values
            self.y = y_mapped.values
            self.classes = list(class_mapping.values())
            print("Data preprocessing completed.")
        except Exception as e:
            print("Error during data preprocessing:", e)

    def select_best_parameters(self):
        """
        Use GridSearchCV to find the best hyperparameters for the XGBoost model.
        """
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            param_grid = {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }

            grid_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)

            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters found: {self.best_params}")
        except Exception as e:
            print("Error during parameter selection:", e)

    def train_model(self):
        """
        Train the XGBoost model using the best parameters found.
        """
        try:
            if not self.best_model:
                raise ValueError("No best model found. Run select_best_parameters() first.")
            
            self.best_model.fit(self.X_train, self.y_train)

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
        Predicts the class for new data using the trained XGBoost model.
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
Classification Report on Test Data:
              precision    recall  f1-score   support

           0       0.58      1.00      0.74         7
           1       0.65      0.60      0.62        40
           2       0.60      0.56      0.58        27
           3       0.27      0.43      0.33         7
           4       0.50      0.33      0.40         6
           5       1.00      0.33      0.50         3

    accuracy                           0.58        90
   macro avg       0.60      0.54      0.53        90
weighted avg       0.60      0.58      0.58        90


Confusion Matrix:
[[ 7  0  0  0  0  0]
 [ 5 24  7  2  2  0]
 [ 0  9 15  3  0  0]
 [ 0  3  1  3  0  0]
 [ 0  1  1  2  2  0]
 [ 0  0  1  1  0  1]]
Predictions for new data: [1 0 1]"""



if __name__ == "__main__":
    file_path = 'filtered_output.csv'  
    feature_columns = ['queue1', 'queue2']  
    target_column = 'action' 

    xgb_classifier = XGBoostClassifier(filepath=file_path, feature_columns=feature_columns, target_column=target_column)

    xgb_classifier.load_data_as_pd_dataframe()

    xgb_classifier.pre_process_data()

    xgb_classifier.select_best_parameters()

    xgb_classifier.train_model()


