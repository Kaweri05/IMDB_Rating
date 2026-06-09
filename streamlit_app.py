import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available (for deployment, you might pre-download these)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the pre-trained model and vectorizer
try:
    model = joblib.load('linear_svc_tfidf_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    st.success("Model and components loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or vectorizer files not found. Please ensure 'linear_svc_tfidf_model.pkl', 'tfidf_vectorizer.pkl', and 'label_encoder.pkl' are in the same directory.")
    st.stop()

# Initialize lemmatizer and stopwords for preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 2. Remove special characters and numbers, keep only letters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # 3. Convert to lowercase
    text = text.lower()
    # 4. Tokenization
    tokens = text.split()
    # 5. Remove stopwords and 6. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below to get its sentiment (positive/negative).")

user_input = st.text_area("Enter your movie review here:", "This movie was absolutely fantastic! I loved every minute of it.")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        cleaned_input = preprocess_text(user_input)
        
        # Vectorize the cleaned text
        input_vectorized = vectorizer.transform([cleaned_input])
        
        # Make prediction
        prediction_encoded = model.predict(input_vectorized)
        
        # Decode the prediction
    
        prediction_encoded = model.predict(input_vectorized)

st.write("Encoded Prediction:", prediction_encoded)

prediction_sentiment = label_encoder.inverse_transform(prediction_encoded)[0]

st.write("Decoded Prediction:", prediction_sentiment)
        
        st.subheader("Analysis Result:")
        if prediction_sentiment == 'positive':
            st.success(f"Sentiment: **{prediction_sentiment.upper()}** 😊")
        else:
            st.error(f"Sentiment: **{prediction_sentiment.upper()}** 😠")
    else:
        st.warning("Please enter some text to analyze.")
