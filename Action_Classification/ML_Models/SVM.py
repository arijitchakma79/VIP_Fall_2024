import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class SVM:
    def __init__(self, filepath, feature_columns, target_column):
        self.filepath = filepath
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.y_pred = None

    def load_data_into_pd(self):
        """Loads data from the CSV file."""
        self.data = pd.read_csv(self.filepath)

    def extract_features_and_target(self):
        """Extracts features and target from the data."""
        self.X = self.data[self.feature_columns].values
        self.y = self.data[self.target_column].values

    def split_data_into_training_and_testing_set(self):
        """Splits the data into training and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_model_with_best_params(self):
        """Trains the SVM model using the best parameters found previously."""
        # Use the best parameters directly
        self.best_model = SVC(C=0.1, kernel='poly', degree=3, gamma='auto')
        
        # Fit the model to the training data
        self.best_model.fit(self.X_train, self.y_train)
        
        # Make predictions on the test set
        self.y_pred = self.best_model.predict(self.X_test)
        
        # Print evaluation metrics
        print("\nClassification Report on Test Data:")
        print(classification_report(self.y_test, self.y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))

    def save_test_set_with_predictions(self, filename="test_set_with_predictions.csv"):
        """Saves the test set (X_test, y_test) along with predictions to a CSV file."""
        if self.y_pred is None:
            raise ValueError("No predictions available. Please train the model and generate predictions before saving.")

        # Create a DataFrame for X_test with feature column names
        test_data = pd.DataFrame(self.X_test, columns=self.feature_columns)
        test_data[self.target_column] = self.y_test  # Add true labels
        test_data['predictions'] = self.y_pred       # Add predictions from the best model
        test_data.to_csv(filename, index=False)
        print(f"Test set with predictions saved to {filename}")

    def predict(self, new_data):
        """Predicts the class for new data using the best model."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Please train the model before making predictions.")
        
        predictions = self.best_model.predict(new_data)
        return predictions

if __name__ == "__main__":
    svm = SVM(filepath='../Files/combined.csv', feature_columns=['queue1', 'queue2'], target_column='action')

    svm.load_data_into_pd()
    svm.extract_features_and_target()
    svm.split_data_into_training_and_testing_set()
    svm.train_model_with_best_params()

    # Save the testing set along with predictions to a CSV file
    svm.save_test_set_with_predictions(filename="test_set_with_predictions.csv")

    # Predict on new data (example input; replace with actual data)
    new_data = np.array([[15, 30], [5, 40], [35, 20]])  
    predictions = svm.predict(new_data)
    print("Predictions for new data:", predictions)
