import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    # Load your dataset here
    data = pd.read_csv('data/your_dataset.csv')
    return data

def preprocess_data(data):
    # Preprocess your data here
    # Example: Handle missing values, encode categorical variables, etc.
    data = data.dropna()
    return data

if __name__ == '__main__':
    data = load_data()
    processed_data = preprocess_data(data)
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
