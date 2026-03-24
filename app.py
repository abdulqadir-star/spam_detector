import streamlit as st
import pickle
import string
from string import punctuation

from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

ps = PorterStemmer()
stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])


def datatransform(text):
# 1 convert lower case
    text=text.lower() 
# 2 divide small parts
    # text=nltk.word_tokenize(text)
    text = text.split()
# 3 special characters
    b=[]
    for n in text:
        if n.isalnum():
            b.append(n)
# 4 stop words & punctuation
    text=b[:]
    b.clear()
    for n in text:
        if n not in stopwords and n not in string.punctuation:
            b.append(n)
# 5 staming
    text=b[:]
    b.clear()
    for n in text:
        b.append(ps.stem(n))
        
    return " ".join(b)

# st.title('how are you')
tfidf = pickle.load(open('vectorize.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))



# st.markdown(html_temp, unsafe_allow_html=True)

input_sms = st.text_area("Enter the message")

if st.button('Inquire'):

    # 1. preprocess
    transformed_sms = datatransform(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header("Spam")
    else:
        st.header("Not Spam")

