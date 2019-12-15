import pandas as pd
from joblib import dump

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

data = pd.read_csv("data/extremely_clean_dataset.csv")
data = data[:48000]

X = data["review"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=72
)

vectorizer = CountVectorizer(ngram_range=(1, 2))

classifier = LogisticRegression(
    C=0.23357214690901212,
    l1_ratio=None,
    multi_class="auto",
    n_jobs=None,
    penalty="l2",
    solver="liblinear",
    tol=0.0001,
    verbose=0,
    warm_start=False,
)

pipe = Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

pipe.fit(X_train, y_train)
print("Test set accuracy is: " + str(pipe.score(X_test, y_test)))

# Save model
dump(pipe, "model.joblib")
