import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Loading the dataset
df = pd.read_csv("C:/Users/karth/fakenews-app/fake_or_real_news.csv") 
texts = df['text'].astype(str).values
labels = (df['label'] == 'FAKE').astype(int).values

# Tokenize
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=300)
y = labels

# LSTM Model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=300),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=3, batch_size=128, validation_split=0.2)

# Save
model.save("lstm_fakenews_model.h5")
joblib.dump(tokenizer, "tokenizer.pkl")

print("Model and tokenizer saved.")
