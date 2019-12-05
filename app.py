import joblib

from flask import Flask, request, render_template

from clean import tokenizer

app = Flask(__name__)

# Import trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    prediction = model.predict([review])

    is_positive = prediction[0] == "positive"

    return render_template("index.html", is_positive=is_positive)


if __name__ == "__main__":
    app.run()
