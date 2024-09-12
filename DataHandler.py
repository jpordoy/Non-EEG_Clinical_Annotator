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