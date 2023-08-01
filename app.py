import streamlit as st
import time
import pandas  as pd
import pickle
import nltk
import spacy
import numpy as np



def preprocess():

#Preprocessing steps

    #Lowercasing

    # df['text'] = df['text'].str.lower()



    # Removing HTML tags

    import re

    def remove_html(text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)

    df['text'] = df['text'].apply(remove_html)



    #Removing Contradictions
    
    nlp = spacy.load('en_core_web_sm')


    import contractions

    def remove_contradictions(text):

      return " ".join([contractions.fix(word.text) for word in nlp(text)])

    df['text'] = df['text'].apply(remove_contradictions)



    #Removing URL

    import re

    def remove_url(text):
        pattern = re.compile(r'https?://\S+|www\.\S+')
        return pattern.sub(r'', text)

    df['text'] = df['text'].apply(remove_url)



    #Remmove punctuation

    import string

    punc = string.punctuation

    def  remove_punc(text):

        return text.translate(str.maketrans('', '', punc))

    df['text'] = df['text'].apply(remove_punc)


    from autocorrect import Speller

    check = Speller()

    def check_spell(text):

      return check(text)

    df['text'] = df['text'].apply(check_spell)


    # Removing stop words


    from nltk.corpus import stopwords

    stopwords = stopwords.words('english')

    def remove_stop_words(text):
        ls = []
        new = []

        ls = nlp(text)

        for word in ls:
            if word.text not in stopwords:

                new.append(word.text)

        return ' '.join(new)

    df['text'] = df['text'].apply(remove_stop_words)



    # Lemmetization

#     from nltk.stem.wordnet import WordNetLemmatizer

#     lemmatizer = WordNetLemmatizer()

#     def Lemmetization(text):

#       return " ".join([lemmatizer.lemmatize(word.text) for word in nlp(text)])

#     df['text'] = df['text'].apply(Lemmetization)
    
    
    #Stemming
    
    from nltk.stem.porter import PorterStemmer
    
    ps = PorterStemmer()
    
    def Stemming(text):

      return " ".join([ps.stem(word.text) for word in nlp(text)])

    df['text'] = df['text'].apply(Stemming)


#Title
st.title("SMS Spam Classifier Web App")


#Message

msg = st.text_input('Type your message below')

#Preprocess

df = pd.DataFrame()

df['text'] = [msg]
print(df)
preprocess()
print(df)

#Precict
def answer():
    
    pipeline = pickle.load(open('model_pipeline_svc.pkl', 'rb'))
    prediction = pipeline.predict(df.values.tolist()[0])[0]
    print(prediction, pipeline.predict_proba(df.values.tolist()[0]))
    
    if prediction == 0:
        return False
    else:
        return True



    
#Creating a button for prediction:

if st.button('Predict'):
    
    # Show and update progress bar
    
    bar = st.progress(50)
    
    value = answer()
    
#     if st.checkbox('Show your input text'):
#         st.write(msg)
#     else:
#         pass
    if value == False:
       
        st.markdown("<h1 style='text-align: center; color: green;'>NOT Spam</h1>", unsafe_allow_html=True)
        
    else:
        
        st.markdown("<h1 style='text-align: center; color: red;'>Spam</h1>", unsafe_allow_html=True)
    
    df.drop(columns='text')
    
    time.sleep(0.5)
    bar.progress(100)






