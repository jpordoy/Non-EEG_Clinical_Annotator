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