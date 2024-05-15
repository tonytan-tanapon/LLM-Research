import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from model import build_model

def load_data():
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    return X_train, y_train

def train_model():
    X_train, y_train = load_data()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = build_model(input_shape=(X_train_scaled.shape[1],))
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32)
    model.save('results/model.h5')

if __name__ == '__main__':
    train_model()
