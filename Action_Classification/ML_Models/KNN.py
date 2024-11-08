import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class KNN_Algorithm:
    def __init__(self, filepath, feature_columns, target_column):
        self.file_path = filepath
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.data = None
        self.X = None
        self.y = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.classes = None
        self.best_model = None
        self.best_k = None
        self.train_scores = []
        self.test_scores = []
        self.best_k_accuracy = 0

    def load_data_as_pd_dataframe(self):
        """
        Load data from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading the file: {e}")

    def pre_process_data(self):
        """
        Preprocess the data: handle missing values and prepare features and target.
        """
        try:
            X = self.data[self.feature_columns].copy()  # Avoid SettingWithCopyWarning
            y = self.data[self.target_column]

            # Fill missing values in numerical columns with the mean
            numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
            X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].mean())

            # Get unique class names for the target
            class_names = sorted(np.unique(y).astype(str))
            self.X = X
            self.y = y
            self.classes = class_names
            print("Data preprocessing completed.")
        except Exception as e:
            print("Error during data preprocessing:", e)

    def plot_data(self):
        """
        Plot pairwise scatter plots for features with class distinction.
        """
        try:
            sns.pairplot(data=self.X.assign(target=self.y), hue='target', palette='Set1', diag_kind=None, corner=True)
            plt.suptitle("Pairwise Scatter Plots with Class Distinction", y=1.02)
            plt.show()
        except Exception as e:
            print(f"Error plotting data: {e}")

    def select_best_k(self):
        """
        Split data, scale it, and find the best K value for KNN.
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
        Train the KNN model using the best K value.
        """
        try:
            self.best_model = KNeighborsClassifier(n_neighbors=self.best_k)
            self.best_model.fit(self.X_train_scaled, self.y_train)
            self.best_k_accuracy = self.best_model.score(self.X_test_scaled, self.y_test)
            print(f"Model trained successfully. Test accuracy: {self.best_k_accuracy:.2f}")
        except Exception as e:
            print("Error during model training:", e)

    def predict(self, X):
        """
        Predict using the trained KNN model.
        """
        try:
            # Convert DataFrame to NumPy array if needed
            if isinstance(X, pd.DataFrame):
                X = X.values  # Convert to a NumPy array without feature names

            return self.best_model.predict(X)
        except Exception as e:
            print("Error during prediction:", e)
            return None

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the test predictions.
        """
        try:
            self.y_pred = self.best_model.predict(self.X_test_scaled)
            cm = confusion_matrix(self.y_test, self.y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.classes,
                        yticklabels=self.classes)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.show()
        except Exception as e:
            print("Error plotting confusion matrix:", e)

def main():
    # Specify file path and columns
    file_path = '../Files/combined.csv'  # Replace with your CSV file path
    feature_columns = ['queue1', 'queue2']  # Adjust as needed based on your dataset
    target_column = 'action'  # Replace with your target column name

    # Instantiate the KNN_Algorithm class
    knn_algo = KNN_Algorithm(filepath=file_path, feature_columns=feature_columns, target_column=target_column)

    # Step 1: Load the data
    knn_algo.load_data_as_pd_dataframe()

    # Step 2: Preprocess the data
    knn_algo.pre_process_data()

    # Step 3: (Optional) Plot data for visualization
    knn_algo.plot_data()

    # Step 4: Find the best K value
    knn_algo.select_best_k()

    # Step 5: Train the model
    knn_algo.train_model()

    # Step 6: Print the accuracy of the best model
    print(f"\nBest model test accuracy: {knn_algo.best_k_accuracy:.2f}")

    # Step 7: Plot the confusion matrix to evaluate the model's performance
    knn_algo.plot_confusion_matrix()

    # Step 8: Example prediction (optional)
    # Replace [value1, value2] with an actual test input
    example_input = pd.DataFrame([[10, 20]], columns=feature_columns)  # Example input for prediction
    prediction = knn_algo.predict(example_input)  # Pass as DataFrame, conversion happens in predict()
    print("\nPredicted class for the example input:", prediction)

if __name__ == "__main__":
    main()
