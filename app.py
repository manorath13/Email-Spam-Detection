import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn

stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def transform_text(text):
    #lowering the text
    text = text.lower()
    #removing punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    #stemming the words
    text = [stemmer.stem(word) for word in text if word not in stopwords_set]
    #joing the words into a string
    text = ' '.join(text)
    return text

vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('mnb_model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_mail = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_email = transform_text(input_mail)
    # 2. vectorize
    vector_input = vectorizer.transform([transformed_email])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# test with examples.txt