import pandas as pd
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from clean import tokenizer

data = pd.read_csv("data/imdb_dataset.csv")

# Keep the first 500 elements to reduce build time
data = data[:3000]

X = data["review"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=72
)

classifier = LogisticRegression(solver="lbfgs")

vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 2))

# Create pipeline using Bag of Words
pipe = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

# Fit Model
pipe.fit(X_train, y_train)
print("Test set accuracy is: " + str(pipe.score(X_test, y_test)))

# Save as file
joblib.dump(pipe, "model.pkl")
