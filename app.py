# import streamlit as st
# import pickle
# import string
# from string import punctuation

# from nltk.tokenize import word_tokenize
# import nltk
# from nltk.stem.porter import PorterStemmer
# import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')

# ps = PorterStemmer()
# stopwords = set([
#     'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
#     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
#     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
#     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
#     'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
#     'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
#     'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
#     'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
#     'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
# ])


# def datatransform(text):
# # 1 convert lower case
#     text=text.lower() 
# # 2 divide small parts
#     # text=nltk.word_tokenize(text)
#     text = text.split()
# # 3 special characters
#     b=[]
#     for n in text:
#         if n.isalnum():
#             b.append(n)
# # 4 stop words & punctuation
#     text=b[:]
#     b.clear()
#     for n in text:
#         if n not in stopwords and n not in string.punctuation:
#             b.append(n)
# # 5 staming
#     text=b[:]
#     b.clear()
#     for n in text:
#         b.append(ps.stem(n))
        
#     return " ".join(b)

# # st.title('how are you')
# tfidf = pickle.load(open('vectorize.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))



# # st.markdown(html_temp, unsafe_allow_html=True)

# input_sms = st.text_area("Enter the message")

# if st.button('Inquire'):

#     # 1. preprocess
#     transformed_sms = datatransform(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 0:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")








import streamlit as st
import pickle
import string
from nltk.stem.porter import PorterStemmer

# ---------------------- BASIC SETUP ----------------------
st.set_page_config(page_title="Spam Detector", layout="centered")

# ---------------------- CSS STYLING ----------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    color: #333;
}

.input-box textarea {
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
    padding: 10px !important;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #45a049;
}

.result {
    padding: 15px;
    border-radius: 10px;
    margin-top: 20px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}

.spam {
    background-color: #ff4d4d;
    color: white;
}

.not-spam {
    background-color: #4CAF50;
    color: white;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<h1 class='title'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)

# ---------------------- NLP SETUP ----------------------
ps = PorterStemmer()

stopwords = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves',
    'he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their',
    'theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for','with','about',
    'against','between','into','through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then','once','here','there','when',
    'where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor',
    'not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now'
])

# ---------------------- TEXT PROCESSING ----------------------
def datatransform(text):
    text = text.lower()
    words = text.split()

    filtered_words = []
    for word in words:
        if word.isalnum() and word not in stopwords and word not in string.punctuation:
            filtered_words.append(ps.stem(word))

    return " ".join(filtered_words)

# ---------------------- LOAD MODELS ----------------------
with open('vectorize.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------- INPUT ----------------------
input_sms = st.text_area("Enter your message here")

# ---------------------- PREDICTION ----------------------
if st.button("Check Message"):

    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        transformed_sms = datatransform(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 0:
            st.markdown('<div class="result spam">🚨 Spam Message</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result not-spam">✅ Not Spam</div>', unsafe_allow_html=True)

# ---------------------- FOOTER ----------------------
st.markdown("<div class='footer'>Developed by Ansari</div>", unsafe_allow_html=True)
