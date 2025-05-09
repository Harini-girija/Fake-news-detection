import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('/Train.csv')

# Display dataset shape
print("Dataset shape:", df.shape)

# Display first few rows
print("\nSample rows:\n", df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Fill missing values
df = df.fillna('')

# Get actual column names
print("\nColumn names:\n", df.columns)

# Features and labels (Updated based on actual column names)
texts = df['text']  
labels = df['label']  

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=7)

# Train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n✅ Accuracy: {round(accuracy * 100, 2)}%")
print("Confusion Matrix:\n", conf_matrix)
