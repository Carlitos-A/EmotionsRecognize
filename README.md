# EmotionsRecognize 💬

App en **Streamlit** que detecta la emoción transmitida por un texto en
**español**: escribes una frase y el modelo predice si transmite alegría,
tristeza, enojo, etc. (~75% de accuracy en el conjunto de evaluación).

## Cómo funciona

```text
Texto en español
   │  deep-translator (GoogleTranslator)
   ▼
Texto en inglés  ← el modelo está entrenado en inglés
   │  limpieza: minúsculas, sin puntuación, sin stopwords (NLTK)
   ▼
CountVectorizer (bag of words)
   │
   ▼
Multinomial Naive Bayes (modelo_emociones.pkl)
   │
   ▼
Emoción en inglés → traducida de vuelta al español
```

## ⚖️ La decisión técnica importante

El dataset de entrenamiento disponible está en **inglés**, pero el caso de uso
es **español**. En lugar de entrenar un modelo peor con datos escasos en
español, se optó por traducir la entrada y la salida con GoogleTranslator:

- ✅ Aprovecha un modelo entrenado con un corpus grande y variado.
- ❌ Añade latencia (2 llamadas de traducción por predicción) y dependencia de
  un servicio externo.
- ❌ La traducción puede introducir ruido: matices como sarcasmo o expresiones
  locales se degradan.

Es un trade-off explícito: **accuracy y simplicidad del modelo a cambio de
latencia y dependencia externa**. Un modelo entrenado nativamente en español
sería el paso siguiente.

## 🚀 Stack

- **Streamlit** — interfaz web
- **scikit-learn** — CountVectorizer + MultinomialNB
- **NLTK** — stopwords y limpieza de texto
- **deep-translator** — traducción es→en / en→es
- **joblib** — serialización del modelo y vectorizador

## Ejecutar

```bash
cd EmotionsApp
pip install -r requirements.txt
streamlit run app.py
```

El repo incluye `.devcontainer` para desarrollar en VS Code sin instalar nada
localmente.

## 🔁 Reproducibilidad y reentrenamiento

Los artefactos versionados (`modelo_emociones.pkl`, `vectorizer.pkl`) fueron
generados con el mismo pipeline que usa la app. Para reentrenarlos con datos
nuevos (script de referencia):

```python
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from app import limpiar_texto  # misma limpieza que usa la app

# textos: lista de textos en inglés; etiquetas: emoción de cada texto
textos_limpios = [limpiar_texto(t) for t in textos]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos_limpios)

modelo = MultinomialNB()
modelo.fit(X, etiquetas)

joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(modelo, "modelo_emociones.pkl")
```

> Pendiente: subir el script/notebook original de entrenamiento con el
> dataset usado y la matriz de confusión por clase.
