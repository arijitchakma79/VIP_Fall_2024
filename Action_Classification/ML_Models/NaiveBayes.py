import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from Action_Classifier import Action_Classifier


class NaiveBayes(Action_Classifier):
    def select_best_parameters(self):
        """
        Naive Bayes does not have hyperparameters to tune in the same way as other models.
        This method is implemented to fulfill the abstract base class requirement.
        """
        print("Naive Bayes does not require parameter tuning. Proceeding with model training...")

    def train_model(self):
        """
        Train the Naive Bayes model.
        """
        try:
            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Initialize the Gaussian Naive Bayes model
            self.best_model = GaussianNB()

            # Train the model
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

    def save_test_set_with_predictions(self, filename="test_set_with_predictions.csv"):
        """
        Saves the test set (X_test, y_test) along with predictions to a CSV file.
        """
        try:
            if self.y_pred is None:
                raise ValueError("No predictions available. Please train the model and generate predictions before saving.")

            # Create a DataFrame for X_test with feature column names
            test_data = pd.DataFrame(self.X_test, columns=self.feature_columns)
            test_data[self.target_column] = self.y_test  # Add true labels
            test_data['predictions'] = self.y_pred       # Add predictions from the model
            test_data.to_csv(filename, index=False)
            print(f"Test set with predictions saved to {filename}")
        except Exception as e:
            print(f"Error saving test set with predictions: {e}")

    def predict(self, new_data):
        """
        Predicts the class for new data using the trained Naive Bayes model.
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
    file_path = 'filtered_output.csv'  # Replace with your file path
    feature_columns = ['queue1', 'queue2']  # Adjust columns based on your dataset
    target_column = 'action'  # Specify your target column

    # Instantiate the NaiveBayes class
    nb_classifier = NaiveBayes(filepath=file_path, feature_columns=feature_columns, target_column=target_column)

    # Load the data
    nb_classifier.load_data_as_pd_dataframe()

    # Preprocess the data and split into training and testing sets
    nb_classifier.pre_process_data()

    # Run the parameter selection method (not needed for Naive Bayes, but for consistency)
    nb_classifier.select_best_parameters()

    # Train the model
    nb_classifier.train_model()

    # Save the test set with predictions
    nb_classifier.save_test_set_with_predictions(filename="test_set_with_predictions.csv")

    # Example prediction
    new_data = np.array([[15, 30], [5, 40], [35, 20]])  # Replace with your actual test data
    predictions = nb_classifier.predict(new_data)
    print("Predictions for new data:", predictions)