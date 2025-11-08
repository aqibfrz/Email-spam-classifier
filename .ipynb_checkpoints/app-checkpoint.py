import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

from PIL import Image

image = Image.open('pngwing.com (2).png')

st.image(image, caption='EMAIL')


page_bg_img = '''
<style>
body {
background-image: url("https://source.unsplash.com/3tYZjGSBwbk");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
