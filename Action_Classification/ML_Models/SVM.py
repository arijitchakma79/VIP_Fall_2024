import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from Action_Classifier import Action_Classifier  

class SVM(Action_Classifier):
    def select_best_parameters(self):
        """
        Use GridSearchCV to find the best hyperparameters for the SVM model.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            # Store the train and test data
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            # Define the parameter grid
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': [ 'poly', 'rbf'],
                'degree': [2, 3, 4],  # Only relevant for 'poly' kernel
                'gamma': ['scale', 'auto']
            }

            # Perform grid search with cross-validation
            grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
            grid_search.fit(self.X_train, self.y_train)

            # Store the best model and parameters
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters found: {self.best_params}")
        except Exception as e:
            print("Error during parameter selection:", e)

    def train_model(self):
        """
        Train the SVM model using the best parameters found.
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
        Predicts the class for new data using the trained SVM model.
        """
        try:
            if self.best_model is None:
                raise ValueError("Model has not been trained yet. Please train the model before making predictions.")
            
            predictions = self.best_model.predict(new_data)
            return predictions
        except Exception as e:
            print("Error during prediction:", e)
            return None

if __name__ == "__main__":
    # Specify the file path and columns
    file_path = '../Files/filtered_output.csv'
    feature_columns = ['queue1', 'queue2']
    target_column = 'action'

    # Instantiate the SVM class
    svm_classifier = SVM(filepath=file_path, feature_columns=feature_columns, target_column=target_column)

    # Load the data
    svm_classifier.load_data_as_pd_dataframe()

    # Preprocess the data
    svm_classifier.pre_process_data()

    # Select the best hyperparameters
    svm_classifier.select_best_parameters()

    # Train the model
    svm_classifier.train_model()

    # Example prediction
    new_data = np.array([[15, 30], [5, 40], [35, 20]])
    predictions = svm_classifier.predict(new_data)
    print("Predictions for new data:", predictions)
