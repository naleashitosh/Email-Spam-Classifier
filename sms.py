import streamlit as st
import pickle as pkl
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

tf = pkl.load(open('tfidf.pkl','rb'))
model = pkl.load(open('model.pkl','rb')) 

ps = PorterStemmer()

def preprocessor(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    ls = []
    for i in text:
        if i.isalnum():
            ls.append(i)
            
    text = ls[:]
    ls.clear()
    
    for i in text:
        if i not in stopwords.words('english') and  i not in string.punctuation:
            ls.append(i)
            
    text = ls[:]
    ls.clear()
    
    for i in text:
        ls.append(ps.stem(i))
        
    return ' '.join(ls)

st.header('SMS spam Classiffier')

sms_text = st.text_area('Enter the SMS')

if st.button('Predict'):
    
    processed_text = preprocessor(sms_text)
    tf_array = tf.transform([processed_text])
    output = model.predict(tf_array)[0]

    if output == 1:
        st.header('SPAM')
    else : st.header('NOT SPAM')