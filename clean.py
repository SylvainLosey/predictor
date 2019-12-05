import pandas as pd
import numpy as np
import re
import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (
    confusion_matrix,
    plot_confusion_matrix,
    average_precision_score,
)
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve

sp = spacy.load("en_core_web_sm")


def to_lower(this_review):
    this_review = this_review.lower()
    return this_review


REMOVE_HTML = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def remove_html(review):
    return REMOVE_HTML.sub(" ", review)


def recognize_it(this_review):
    doc = sp(this_review)

    for i in doc.ents:
        i = str(i)
        this_review = this_review.replace(" " + i, "")
    return this_review


# Implementing lemmatization
def lemmatize_it(this_review):
    filtered_sent = []

    #  "nlp" Object is used to create documents with linguistic annotations.
    lem = sp(this_review)

    # finding lemma for each word
    for word in lem:
        filtered_sent.append(word.lemma_)
    return filtered_sent


def eliminate_stopwords(this_review):
    spacy_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)

    remove_from_stopwordlist = [
        "n't",
        "most",
        "much",
        "never",
        "no",
        "not",
        "nothing",
        "n‘t",
        "n’t",
        "really",
        "top",
        "very",
        "well",
    ]
    for word in spacy_stopwords:
        if word in remove_from_stopwordlist:
            spacy_stopwords.remove(word)

    add_to_stopwords = [
        ".",
        ",",
        "!",
        "?",
        ":",
        "&",
        "...",
        "(",
        ")",
        "-",
        "/",
        '"',
        ";",
        "-PRON-",
        " ",
    ]
    for word in add_to_stopwords:
        spacy_stopwords.append(word)

    filtered_sent = []

    #  "nlp" Object is used to create documents with linguistic annotations.
    doc = this_review

    # filtering stop words
    for word in doc:
        if word not in spacy_stopwords:
            filtered_sent.append(word)
    return filtered_sent


def tokenizer(this_review):
    this_review = to_lower(this_review)
    this_review = remove_html(this_review)
    this_review = recognize_it(this_review)
    this_review = lemmatize_it(this_review)
    this_review = eliminate_stopwords(this_review)
    return this_review
