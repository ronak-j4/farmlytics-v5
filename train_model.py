import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import urllib.request

def train():
    csv_file = 'Crop_recommendation.csv'
    
    # Download dataset if it doesn't exist
    if not os.path.exists(csv_file):
        print("Downloading dataset...")
        url = "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv"
        urllib.request.urlretrieve(url, csv_file)
        print("Downloaded successfully.")

    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    
    # Features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Accuracy: {accuracy * 100:.2f}%")
    
    # Save model
    joblib.dump(model, 'model.pkl')
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train()
