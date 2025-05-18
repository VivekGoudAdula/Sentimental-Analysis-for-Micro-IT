import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import nltk
import re
import string

# Download required nltk data (run this once)
nltk.download('stopwords')
from nltk.corpus import stopwords

# 🧼 Preprocessing function
def clean_text(text):
    text = text.lower()                             # lowercase
    text = re.sub(r'\[.*?\]', '', text)             # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)              # remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # remove punctuation
    text = re.sub(r'\n', '', text)                  # remove newlines
    text = re.sub(r'\w*\d\w*', '', text)            # remove words with numbers
    return text

# 📂 Load your dataset (example: 'sentiment_data.csv' with 'text' and 'label')
df = pd.read_csv('sentiment_data.csv')  # Make sure this CSV exists in the same folder
df['text'] = df['text'].apply(clean_text)

# ✅ Split dataset
X = df['text']
y = df['label']  # Should be 'positive', 'negative', or 'neutral'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔧 Build ML Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('clf', LogisticRegression(solver='liblinear'))
])

# 🚀 Train the model
model.fit(X_train, y_train)

# 📊 Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔍 Test on custom input
while True:
    user_input = input("\nEnter a sentence for sentiment prediction (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    cleaned_input = clean_text(user_input)
    prediction = model.predict([cleaned_input])[0]
    print(f"Predicted Sentiment: {prediction}")
