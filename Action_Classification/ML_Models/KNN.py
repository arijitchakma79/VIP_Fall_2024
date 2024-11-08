from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from Action_Classifier import Action_Classifier
import pandas as pd


class KNN(Action_Classifier):
    def select_best_parameters(self):
        """
        Implement the logic to split data, scale it, and find the best K value for KNN.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            # Store the train and test data
            self.y_train = y_train
            self.y_test = y_test

            # Scale the features
            scaler = StandardScaler()
            self.X_train_scaled = scaler.fit_transform(X_train)
            self.X_test_scaled = scaler.transform(X_test)

            # Reset scores to avoid duplication
            self.train_scores = []
            self.test_scores = []

            # Search for the best K value
            k_range = range(1, 25)
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(self.X_train_scaled, y_train)
                self.train_scores.append(knn.score(self.X_train_scaled, y_train))
                self.test_scores.append(knn.score(self.X_test_scaled, y_test))

            # Plot K optimization
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, self.train_scores, label="Training accuracy")
            plt.plot(k_range, self.test_scores, label='Test accuracy')
            plt.xlabel("K Value")
            plt.ylabel("Accuracy")
            plt.title('Accuracy vs K Value')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Get the best K value based on test scores
            self.best_k = k_range[np.argmax(self.test_scores)]
            print(f"\nBest K value: {self.best_k}")

        except Exception as e:
            print("Error during K selection:", e)

    def train_model(self):
        """
        Implement training the KNN model using the best K value.
        """
        try:
            self.best_model = KNeighborsClassifier(n_neighbors=self.best_k)
            self.best_model.fit(self.X_train_scaled, self.y_train)
            self.best_k_accuracy = self.best_model.score(self.X_test_scaled, self.y_test)
            print(f"Model trained successfully. Test accuracy: {self.best_k_accuracy:.2f}")
        except Exception as e:
            print("Error during model training:", e)


def main():
    file_path = '../Files/filtered_output.csv'  
    feature_columns = ['queue1', 'queue2']  
    target_column = 'action'  

    # Instantiate the KNN class
    knn_algo = KNN(filepath=file_path, feature_columns=feature_columns, target_column=target_column)

    # Load the data
    knn_algo.load_data_as_pd_dataframe()

    # Preprocess the data
    knn_algo.pre_process_data()

    # Find the best K value
    knn_algo.select_best_parameters()

    # Train the model
    knn_algo.train_model()

    # Print the accuracy of the best model
    print(f"\nBest model test accuracy: {knn_algo.best_k_accuracy:.2f}")

    # Plot the confusion matrix to evaluate the model's performance
    knn_algo.plot_confusion_matrix()

    
    
    example_input = pd.DataFrame([[10, 20]], columns=feature_columns)  # Example input for prediction
    prediction = knn_algo.predict(example_input)  # Pass as DataFrame, conversion happens in predict()
    print("\nPredicted class for the example input:", prediction)


if __name__ == "__main__":
    main()