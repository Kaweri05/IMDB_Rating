import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load model files
try:
    model = joblib.load('linear_svc_tfidf_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

except FileNotFoundError:
    st.error(
        "Error: Model or vectorizer files not found. "
        "Please ensure all .pkl files are in the same directory."
    )
    st.stop()

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = text.split()

    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return ' '.join(tokens)


# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write(
    "Enter a movie review below to predict "
    "whether it is positive or negative."
)

user_input = st.text_area(
    "Enter your movie review here:",
    "This movie was absolutely fantastic! I loved every minute of it."
)

# Predict sentiment
if st.button("Analyze Sentiment", key="analyze_btn"):

    if user_input.strip():

        # Preprocess input
        cleaned_input = preprocess_text(user_input)

        # Convert text to vector
        input_vectorized = vectorizer.transform([cleaned_input])

        # Predict
        prediction_encoded = model.predict(input_vectorized)

        # Decode prediction
        prediction_sentiment = label_encoder.inverse_transform(
            prediction_encoded
        )[0]

        st.subheader("Analysis Result:")

        if prediction_sentiment.lower() == "positive":
            st.success(
                f"Sentiment: {prediction_sentiment.upper()} 😊"
            )
        else:
            st.error(
                f"Sentiment: {prediction_sentiment.upper()} 😠"
            )

    else:
        st.warning("Please enter some text to analyze.")
