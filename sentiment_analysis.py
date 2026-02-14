# ===============================
# SOCIAL MEDIA SENTIMENT ANALYSIS
# ===============================

import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
print("Loading dataset...")
data = pd.read_excel("LabeledText.xlsx")

# 2. Basic Inspection
print("\nFirst 5 rows:")
print(data.head())

print("\nChecking missing values:")
print(data.isnull().sum())

# Drop missing text values
data = data.dropna(subset=['Caption'])

# 3. Text Cleaning Function
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)   # Remove URLs
    text = re.sub(r"@\w+", "", text)     # Remove mentions
    text = re.sub(r"#\w+", "", text)     # Remove hashtags
    text = re.sub(r"[^a-zA-Z ]", "", text)  # Remove special characters
    return text.lower()

print("\nCleaning text...")
data['clean_text'] = data['Caption'].apply(clean_text)

# 4. Sentiment Distribution
print("\nSentiment Distribution:")
print(data['LABEL'].value_counts())

data['LABEL'].value_counts().plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show(block=False)

# 5. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['clean_text'])
y = data['LABEL']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Model Training
# 7. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 8. Prediction
predictions = model.predict(X_test)

# 9. Evaluation
print("\nModel Accuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nProcess Completed Successfully.")

#Navie Bayer
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))

#prediction
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)
    return result[0]

print(predict_sentiment("I love this product"))
print(predict_sentiment("This is terrible"))

