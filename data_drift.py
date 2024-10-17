from google.cloud import storage
import pandas as pd
import io  # Import StringIO from io

# Initialize GCS client
storage_client = storage.Client()

# Function to download CSV file from GCS and load into Pandas DataFrame
def download_csv_from_gcs(gcs_uri):
    bucket_name, file_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_text()
    return pd.read_csv(io.StringIO(data))  # Correct StringIO usage

# GCS bucket file paths
BASELINE_FILE_PATH = 'gs://baseline-dataset/hyundi.csv'
NEW_DATA_FILE_PATH = 'gs://baseline-dataset/hyundi_inference.csv'

# Features to monitor for data drift (based on the CSV, excluding 'year' and 'price')
FEATURES_TO_MONITOR = [
    'mileage', 'tax', 'mpg', 'engineSize'
]

# Function to calculate descriptive statistics
def calculate_stats(data, features):
    return {
        'mean': data[features].mean(),
        'std': data[features].std(),
        'iqr': data[features].quantile(0.75) - data[features].quantile(0.25)
    }

# Function to perform data drift analysis
def perform_drift_analysis():
    # Download CSV data
    baseline_data = download_csv_from_gcs(BASELINE_FILE_PATH)
    new_data = download_csv_from_gcs(NEW_DATA_FILE_PATH)

    # Exclude target and year columns ('price' and 'year')
    baseline_data = baseline_data[FEATURES_TO_MONITOR]
    new_data = new_data[FEATURES_TO_MONITOR]

    # Calculate stats
    baseline_stats = calculate_stats(baseline_data, FEATURES_TO_MONITOR)
    new_stats = calculate_stats(new_data, FEATURES_TO_MONITOR)

    # Thresholds for drift detection
    mean_threshold = 0.1
    std_threshold = 0.2
    iqr_threshold = 0.15

    # Compare stats and store drift results
    drift_report = []
    for feature in FEATURES_TO_MONITOR:
        mean_diff = abs(new_stats['mean'][feature] - baseline_stats['mean'][feature]) / baseline_stats['mean'][feature]
        std_diff = abs(new_stats['std'][feature] - baseline_stats['std'][feature]) / baseline_stats['std'][feature]
        iqr_diff = abs(new_stats['iqr'][feature] - baseline_stats['iqr'][feature]) / baseline_stats['iqr'][feature]

        feature_report = f"Feature: {feature}\n"
        if mean_diff > mean_threshold:
            feature_report += f"Mean drift detected: {mean_diff:.2%}\n"
        if std_diff > std_threshold:
            feature_report += f"Standard deviation drift detected: {std_diff:.2%}\n"
        if iqr_diff > iqr_threshold:
            feature_report += f"IQR drift detected: {iqr_diff:.2%}\n"
        drift_report.append(feature_report)

    return "\n\n".join(drift_report)

# Run the drift analysis
if __name__ == "__main__":
    drift_report = perform_drift_analysis()
    print("Drift Analysis Report:")
    print(drift_report)

