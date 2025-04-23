import os
import streamlit as st
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
#Pasar a Español
from deep_translator import GoogleTranslator
def traducir(texto, origen, destino):
    return GoogleTranslator(source=origen, target=destino).translate(texto)

# Descargar recursos de NLTK
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Función para limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = "".join([c for c in texto if c not in string.punctuation])
    palabras = texto.split()
    palabras = [word for word in palabras if word not in stop_words]
    return " ".join(palabras)

# Cargar modelo y vectorizador entrenados
modelo = joblib.load("modelo_emociones.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Interfaz de Streamlit
st.set_page_config(page_title="Detector de Emociones", page_icon="💬")
st.title("🔮 Detector de Emociones por Texto")
st.write("Escribí una frase y te diré qué emoción transmite.")

frase = st.text_input("Tu frase aquí:")
input_text = traducir(frase, 'es', 'en')



if input_text:
    texto_limpio = limpiar_texto(input_text)
    texto_vect = vectorizer.transform([texto_limpio])
    prediccion = modelo.predict(texto_vect)
    st.subheader("💬 Emoción detectada:")
    respuesta_traducida = traducir(prediccion[0], 'en', 'es')
    st.success(respuesta_traducida)
