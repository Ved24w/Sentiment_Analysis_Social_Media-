import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Assume 'processed/cleaned_data.csv' is ready with 'text' and 'sentiment' columns
DATA_PATH = 'data/processed/cleaned_data.csv'
MODEL_PATH = 'results/trained_model.pkl'

def train_sentiment_model():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure sentiment is numerical if needed, e.g., 0=Negative, 1=Neutral, 2=Positive
        df['sentiment'] = df['sentiment'].astype('category').cat.codes
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Please ensure you have run the data collection and cleaning steps.")
        return

    # 2. Split Data
    X = df['text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Feature Extraction (TF-IDF Vectorization)
    print("Fitting TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # 4. Model Training (Using Logistic Regression as a solid baseline)
    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_vectorized, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test_vectorized)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    
    # 6. Save Model and Vectorizer
    os.makedirs('results', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, 'results/tfidf_vectorizer.pkl') # Save vectorizer for future use
    print(f"\nModel and Vectorizer successfully saved to {os.path.abspath('results/')}")

if __name__ == '__main__':
    train_sentiment_model()
