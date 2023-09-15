import pandas as pd

# Sample data for text classification
data = {
    'text': [
        "This is a positive example.",
        "Another positive sentence.",
        "A negative example here.",
        "Yet another negative statement."
    ],
    'label': [1, 1, 0, 0]  # 1 for positive, 0 for negative
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'text_classification_dataset.csv'
df.to_csv(csv_file_path, index=False)

print(f'Dataset saved to {csv_file_path}')
