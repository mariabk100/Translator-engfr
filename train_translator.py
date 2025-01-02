import pandas as pd
import regex as re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pickle

# Load and preprocess data
df = pd.concat([pd.read_csv('/content/en.csv', header=None), pd.read_csv('/content/fr.csv', header=None)], axis=1)
df.columns = ['English', 'French']
df['English'] = df['English'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
df['French'] = df['French'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))

# Tokenize English
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(df['English'])
eng_sequences = eng_tokenizer.texts_to_sequences(df['English'])

# Tokenize French
fr_tokenizer = Tokenizer()
fr_tokenizer.fit_on_texts(df['French'])
fr_sequences = fr_tokenizer.texts_to_sequences(df['French'])

# Save tokenizers
with open('eng_tokenizer.pkl', 'wb') as f:
    pickle.dump(eng_tokenizer, f)
with open('fr_tokenizer.pkl', 'wb') as f:
    pickle.dump(fr_tokenizer, f)

# Pad sequences
max_english_length = max(len(seq) for seq in eng_sequences)
max_french_length = max(len(seq) for seq in fr_sequences)
eng_padded = pad_sequences(eng_sequences, maxlen=max_english_length, padding='post')
fr_padded = pad_sequences(fr_sequences, maxlen=max_french_length, padding='post')

# Model architecture
input_shape = (max_english_length,)
encoder_inputs = Input(shape=input_shape, name='encoder_inputs')
encoder_embedding = Embedding(len(eng_tokenizer.word_index) + 1, 128, input_length=max_english_length)(encoder_inputs)
encoder_outputs, encoder_state_h, encoder_state_c = LSTM(128, return_state=True)(encoder_embedding)
decoder_inputs = RepeatVector(max_french_length)(encoder_state_h)
decoder_lstm = LSTM(128, return_sequences=True)(decoder_inputs)
decoder_dense = TimeDistributed(Dense(len(fr_tokenizer.word_index) + 1, activation='softmax'))
decoder_outputs = decoder_dense(decoder_lstm)

model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split data
eng_train, eng_test, fr_train, fr_test = train_test_split(eng_padded, fr_padded, test_size=0.2, random_state=42)

# Train the model
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001)
model.fit(eng_train, fr_train, validation_data=(eng_test, fr_test), epochs=10, batch_size=32, callbacks=[early_stopping])

# Save the model
model.save('translator_model.h5')


