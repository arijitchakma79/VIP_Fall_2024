import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Action_Classifier(ABC):
    def __init__(self, filepath, feature_columns, target_column):
        """
        Initialize the Action_Classifier with the given data path, features, and target.
        """
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
        Preprocess the data by handling missing values and extracting features and target.
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

    @abstractmethod
    def select_best_parameters(self):
        """
        Abstract method to be implemented for selecting the best K value.
        """
        pass

    @abstractmethod
    def train_model(self):
        """
        Abstract method to be implemented for training the classifier model.
        """
        pass

    def predict(self, X):
        """
        Predict using the trained model.
        """
        try:
            # Convert DataFrame to NumPy array if needed
            if isinstance(X, pd.DataFrame):
                X = X.values  # Convert to a NumPy array without feature names

            return self.best_model.predict(X)
        except Exception as e:
            print("Error during prediction:", e)
            return None
        
    def show_confusion_matrix(self, y_test, y_pred):
        print("Classification")
        print(classification_report(y_test, y_pred))

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the test predictions.
        """
        try:
            self.y_pred = self.best_model.predict(self.X_test_scaled)
            cm = confusion_matrix(self.y_test, self.y_pred)
            print("Confusion Matrix:")
            print(cm)
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            
            tick_marks = np.arange(len(self.classes))
            plt.xticks(tick_marks, self.classes, rotation=45, ha='right')
            plt.yticks(tick_marks, self.classes)

           
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='black')

            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print("Error plotting confusion matrix:", e)

