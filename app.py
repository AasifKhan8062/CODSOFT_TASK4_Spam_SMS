import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# NLTK Setup
# -----------------------------

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()

# -----------------------------
# Text Preprocessing
# -----------------------------

def testModify(x):

    x = x.lower()

    x = nltk.word_tokenize(x)

    y = []

    for i in x:
        if i.isalnum():
            y.append(i)

    x = y.copy()

    y.clear()

    for i in x:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    x = y.copy()

    y.clear()

    for i in x:
        y.append(ps.stem(i))

    return " ".join(y)


# -----------------------------
# Load TF-IDF
# -----------------------------

with open("tfidfvectorizer.pkl", "rb") as f:
    tfd = pickle.load(f)


# -----------------------------
# Load Model
# -----------------------------

with open("MultinomialNBModel.pkl", "rb") as f:
    MNB = pickle.load(f)


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("📩 Spam Message Detector")

st.write(
    "Enter a message below to check whether it is **Spam** or **Not Spam**."
)

st.divider()


# -----------------------------
# Input
# -----------------------------

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a prize...",
    height=150
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message.")

    else:

        # Text preprocessing
        cleaned_message = testModify(message)

        # TF-IDF transformation
        message_vector = tfd.transform([cleaned_message])

        # Prediction
        prediction = MNB.predict(message_vector)[0]

        # Probability
        if hasattr(MNB, "predict_proba"):
            probability = MNB.predict_proba(message_vector)[0]
            confidence = max(probability) * 100
        else:
            confidence = None

        # -----------------------------
        # Result
        # -----------------------------

        if prediction == 1 or str(prediction).lower() == "spam":

            st.error("🚨 SPAM MESSAGE")

            if confidence is not None:
                st.write(
                    f"Confidence: **{confidence:.2f}%**"
                )

        else:

            st.success("✅ NOT SPAM")

            if confidence is not None:
                st.write(
                    f"Confidence: **{confidence:.2f}%**"
                )


# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "Machine Learning Spam Detection | TF-IDF + Multinomial Naive Bayes"
)