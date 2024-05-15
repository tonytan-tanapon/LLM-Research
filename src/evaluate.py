import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    test_data = pd.read_csv('data/test_data.csv')
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    return X_test, y_test

def evaluate_model():
    X_test, y_test = load_data()
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    model = tf.keras.models.load_model('results/model.h5')
    predictions = model.predict(X_test_scaled)
    y_pred = (predictions > 0.5).astype(int)
    
    report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
    matrix = confusion_matrix(y_test, y_pred)
    
    with open('results/evaluation_results.txt', 'w') as f:
        f.write('Model Evaluation Results\n')
        f.write('='*24 + '\n')
        f.write(report + '\n')
        f.write('Confusion Matrix:\n')
        f.write(str(matrix) + '\n')
    
    print('Evaluation results saved to results/evaluation_results.txt')

if __name__ == '__main__':
    evaluate_model()
