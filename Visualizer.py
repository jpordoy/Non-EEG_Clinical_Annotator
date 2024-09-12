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
