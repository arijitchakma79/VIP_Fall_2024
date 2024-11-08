import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


class Action_Classifier:
    def __init__(self, filepath, feature_columns, target_column):
        """
        Initialize the Action_Classifier with the file path, feature columns, and target column.

        Parameters:
        - filepath (str): Path to the CSV file containing the data.
        - feature_columns (list): List of column names to be used as features.
        - target_column (str): Name of the column to be used as the target variable.
        """
        self.filepath = filepath
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.data = None  # To hold the loaded data as a DataFrame
        self.X = None  # To hold the feature matrix
        self.y = None  # To hold the target vector
        self.X_train = None  # Training feature set
        self.X_test = None  # Testing feature set
        self.y_train = None  # Training target set
        self.y_test = None  # Testing target set
        self.best_model = None  # Placeholder for the trained model
        
    def load_data_as_panda_data_frame(self):
        """
        Load the CSV data into a Pandas DataFrame.
        """
        self.data = pd.read_csv(self.filepath)
        print("Data loaded successfully.")
    
    def extract_features_and_target(self):
        """
        Extract features and target variable from the loaded DataFrame.
        """
        self.X = self.data[self.feature_columns].values  
        self.y = self.data[self.target_column].values  
        print("Features and target extracted successfully.")
    
    def split_training_data_into_training_and_testing(self):
        """
        Split the data into training and testing sets.

        The test size is set to 20% of the total data and the random state is fixed for reproducibility.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print("Data split into training and testing sets successfully.")
    
    def visualize_scatter_plot(self):
        """
        Visualize a scatter plot of the first two feature columns, colored by the target variable.
        """
        plt.figure(figsize=(10, 5))
        plt.scatter(self.data[self.feature_columns[0]], self.data[self.feature_columns[1]], 
                    c=self.data[self.target_column], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Action')  
        plt.xlabel(self.feature_columns[0])
        plt.ylabel(self.feature_columns[1])
        plt.title("Scatter Plot of Features Colored by Target Variable")
        plt.show()
    
    def visualize_target_distribution(self):
        """
        Visualize the distribution of the target variable as a bar plot.
        """
        self.data[self.target_column].value_counts().plot(kind='bar', figsize=(8, 5))
        plt.title('Distribution of Target Variable')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.show()
    
    def visualize_feature_histogram_distribution(self):
        """
        Plot histograms for each feature to visualize their distributions.
        """
        self.data[self.feature_columns].hist(bins=30, figsize=(10, 5))
        plt.suptitle('Feature Distribution')
        plt.show()
    
    def visualize_feature_box_plot(self):
        """
        Visualize the distribution of feature columns using box plots.
        """
        self.data[self.feature_columns].plot(kind='box', subplots=True, layout=(1, len(self.feature_columns)), figsize=(10, 5))
        plt.suptitle('Box Plot for Features')
        plt.show()
    
    def show_classification_report(self, y_test, y_pred):
        """
        Print a detailed classification report including precision, recall, F1-score, and support.

        Parameters:
        - y_test (array-like): True labels for the testing set.
        - y_pred (array-like): Predicted labels by the model.
        """
        print("Classification Report")
        print(classification_report(y_test, y_pred))
    
    def show_confusion_matrix(self, y_test, y_pred):
        """
        Display and print the confusion matrix for the model's predictions.

        Parameters:
        - y_test (array-like): True labels for the testing set.
        - y_pred (array-like): Predicted labels by the model.
        """
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
