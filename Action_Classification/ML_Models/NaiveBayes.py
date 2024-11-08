import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
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
        Train the Naive Bayes model and display performance metrics.
        """
        try:
           
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            self.best_model = GaussianNB()
            self.best_model.fit(self.X_train, self.y_train)

            # Make predictions on the test set
            self.y_pred = self.best_model.predict(self.X_test)

            # Calculate and print the accuracy
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print(f"\nModel Accuracy on Test Data: {accuracy:.2f}")

            # Print evaluation metrics
            print("\nClassification Report on Test Data:")
            print(classification_report(self.y_test, self.y_pred))

            # Print and visualize the confusion matrix
            cm = confusion_matrix(self.y_test, self.y_pred)
            print("\nConfusion Matrix:")
            print(cm)

            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(np.arange(len(np.unique(self.y))), np.unique(self.y), rotation=45)
            plt.yticks(np.arange(len(np.unique(self.y))), np.unique(self.y))

            
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='black')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print("Error during model training:", e)

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
    file_path = '../Files/filtered_output.csv'  # location of the input file
    feature_columns = ['queue1', 'queue2']  # feature columns
    target_column = 'action'  # target column

    nb_classifier = NaiveBayes(filepath=file_path, feature_columns=feature_columns, target_column=target_column)            # Instantiate the NaiveBayes class

    nb_classifier.load_data_as_pd_dataframe()       # Load the data

    nb_classifier.pre_process_data()            # Preprocess the data and split into training and testing sets

    nb_classifier.train_model()  # Train the model and display metrics

    
