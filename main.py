import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        """Loads data from a CSV or Excel file."""
        if self.filepath.endswith('.csv'):
            data = pd.read_csv(self.filepath)
        elif self.filepath.endswith('.xlsx'):
            data = pd.read_excel(self.filepath)
        else:
            raise ValueError("Unsupported file format. Supported formats: CSV, Excel")
        return data

    def preprocess_data(self, data):
        """Preprocesses data by handling missing values and scaling."""
        # Handle missing values (e.g., imputation)
        data.fillna(method='ffill', inplace=True)  # Forward fill missing values

        # Scale data using MinMaxScaler (optional)
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        return data_scaled

    def save_data(self, data, filename):
        """Saves processed data to a CSV file."""
        data.to_csv(filename, index=False)

class FeatureExtractor:
    def __init__(self, sample_rate, time_interval):
        self.sample_rate = sample_rate
        self.time_interval = time_interval

    def extract_features(self, data):
        """Extracts statistical features from the data."""
        # Calculate features (e.g., mean, standard deviation, rate of change)
        features = []
        for i in range(0, len(data), self.sample_rate * self.time_interval):
            interval_data = data[i:i + self.sample_rate * self.time_interval]
            features.append([
                np.mean(interval_data),
                np.std(interval_data),
                np.max(interval_data),
                np.min(interval_data),
                np.diff(interval_data).mean() if len(interval_data) > 1 else 0
            ])
        return pd.DataFrame(features, columns=['mean', 'std', 'max', 'min', 'rate_of_change'])

class Analyzer:
    def analyze_features(self, features):
        """Performs analysis on extracted features."""
        # Analyze features (e.g., correlation analysis)
        correlation_matrix = features.corr()
        return correlation_matrix

class Visualizer:
    def visualize_data(self, data, features):
        """Visualizes data and features."""
        # Plot original data
        plt.figure(figsize=(12, 6))
        plt.plot(data)
        plt.title("Original Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()

        # Plot extracted features
        plt.figure(figsize=(12, 6))
        plt.plot(features['mean'], label='Mean')
        plt.plot(features['std'], label='Standard Deviation')
        plt.plot(features['max'], label='Maximum')
        plt.plot(features['min'], label='Minimum')
        plt.plot(features['rate_of_change'], label='Rate of Change')
        plt.title("Extracted Features")
        plt.xlabel("Time Interval")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

class Annotator:
    def __init__(self, data):
        self.data = data

    def provide_annotation_interface(self):
        """Provides a simple annotation interface using Matplotlib."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data)
        plt.title("Annotate Data")
        plt.xlabel("Time")
        plt.ylabel("Value")

        def on_click(event):
            if event.inaxes:
                x, y = event.xdata, event.ydata
                print(f"Clicked at: ({x}, {y})")
                # Add annotation logic here (e.g., store annotation coordinates)

        cid = plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show()