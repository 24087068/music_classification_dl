import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import re
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback

# SETUP:
# SETUP:
class Config:
    SEED = 42
    VAL_SPLIT = 0.20
    METRIC = 'accuracy'
    GENRE_MAP = {'blues': 0, 'country': 1, 'disco': 2, 'hiphop': 3, 'metal': 4, 'pop': 5, 'reggae': 6, 'rock': 7}
    IDX_TO_GENRE = {v: k for k, v in GENRE_MAP.items()}
    if '__file__' in locals():
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        BASE_DIR = os.path.abspath(os.path.join(current_file_dir, '..'))
    else:
        BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
    TRAIN_CSV = os.path.join(BASE_DIR, 'data', 'train.csv')
    TEST_CSV = os.path.join(BASE_DIR, 'data', 'test.csv')
    TRAIN_DIR = os.path.join(BASE_DIR, 'data', 'Train')
    TEST_DIR = os.path.join(BASE_DIR, 'data', 'Test')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
class DatasetManager:
    def __init__(self, config: Config):
        self.cfg = config
        self.df = pd.read_csv(config.TRAIN_CSV)
        self.df['label'] = self.df['genre'].map(config.GENRE_MAP)
        # Stratified split to ensure exact same train/val partition across all models
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.cfg.VAL_SPLIT, random_state=self.cfg.SEED)
        self.train_idx, self.val_idx = next(sss.split(self.df, self.df['label']))
        self.train_df = self.df.iloc[self.train_idx].reset_index(drop=True)
        self.val_df = self.df.iloc[self.val_idx].reset_index(drop=True)


# EDA:
class DataExplorer:
    STOP_WORDS = {
        'the', 'a', 'an', 'i', 'me', 'my', 'and', 'or', 'is', 'it', 'in', 'of', 'to', 'you', 'that', 'this', 'for', 'on', 'are', 'be', 'with', 'as', 'at', 'by', 'we', 'so', 'do', 'not', 'no', 'he', 'she', 'they', 'have', 'had', 'has', 'but', 'from', 'your', 'all', 'was', 'up', 'if', 'out', 'can', 'will', 'get', 'got', 'its', 'been', 'who', 'just', 'im', 's', 'dont', 'oh', 'yeah', 'like', 'now', 'one', 'know', 'want'}
    def __init__(self, config, dataset_manager):
        self.cfg = config
        self.dm = dataset_manager
    def display_audio_samples(self):
        samples = self.dm.df.groupby('genre')['filename'].first().to_dict()
        for genre, fname in samples.items():
            path = os.path.join(self.cfg.TRAIN_DIR, fname.split('.')[0], fname)
            print(genre + ": " + fname)
            ipd.display(ipd.Audio(path))
    def analyze_audio_properties(self):
        samples = self.dm.df.groupby('genre')['filename'].first().to_dict()
        for genre, fname in samples.items():
            path = os.path.join(self.cfg.TRAIN_DIR, fname.split('.')[0], fname)
            y, sr = librosa.load(path, sr=None)
            t = np.arange(len(y)) / sr  # Formula: t_i = i / sfreq
            print("[" + genre + "] sr=" + str(sr) + "Hz, samples=" + str(len(y)) + ", duration=" + str(round(t[-1], 2)) + "s")
    def analyze_lyrics(self):
        df = self.dm.df.copy()
        df['clean'] = df['lyrics'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))
        df['token_len'] = df['clean'].apply(lambda x: len(x.split()))
        genres = sorted(df['genre'].unique())
        # Distribution boxplot
        plt.figure(figsize=(10, 4))
        plt.boxplot([df[df['genre'] == g]['token_len'].values for g in genres], labels=genres)
        plt.title('Lyric Token Length by Genre')
        plt.show()
        # Word frequency counter minus common stop words
        for genre in genres:
            words = ' '.join(df[df['genre'] == genre]['clean']).split()
            filtered = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
            print("[" + genre + "] top-5: " + str(Counter(filtered).most_common(5)))
    def analyze_frequency_domain(self):
        samples = self.dm.df.groupby('genre')['filename'].first().to_dict()
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        for ax, (genre, fname) in zip(axes.flatten(), samples.items()):
            path = os.path.join(self.cfg.TRAIN_DIR, fname.split('.')[0], fname)
            y, sr = librosa.load(path, sr=None)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            ax.set_title(genre)
        plt.show()

# RNN:
N_MELS = 128
SR = 22050
MAX_FRAMES = 650
def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_title(title + ' accuracy')
    ax1.legend()
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_title(title + ' loss')
    ax2.legend()
    plt.show()

def plot_cm(y_true, y_pred, title, labels):
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=labels)
    disp.plot(xticks_rotation=45, colorbar=False)
    plt.title(title)
    plt.show()

class AudioFeatureExtractor:
    def __init__(self, config, n_mels=N_MELS, max_frames=MAX_FRAMES, sr=SR):
        self.cfg = config
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.sr = sr

    def extract(self, filename, is_test=False):
        # Handle train subfolder path schema vs flat test folder
        if is_test:
            path = os.path.join(self.cfg.TEST_DIR, filename)
        else:
            path = os.path.join(self.cfg.TRAIN_DIR, filename.split('.')[0], filename)

        y, sr = librosa.load(path, sr=self.sr, duration=30.0)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, hop_length=1024)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db + 80) / 80
        mel_db = np.clip(mel_db, 0, 1)

        t = mel_db.shape[1]
        if t < self.max_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, self.max_frames - t)))
        else:
            mel_db = mel_db[:, :self.max_frames]
        return mel_db.T

    def build_dataset(self, df, is_test=False):
        arrays = []
        for fn in df['filename']:
            arrays.append(self.extract(fn, is_test=is_test))
        return np.array(arrays)

class AudioGRUModel:
    def __init__(self, config, n_mels=N_MELS, max_frames=MAX_FRAMES):
        self.cfg = config
        self.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'audio_gru_best.keras')

        self.model = keras.Sequential([
            layers.Input(shape=(max_frames, n_mels)),
            layers.MaxPool1D(pool_size=2),
            layers.GRU(64, return_sequences=True),
            layers.GRU(64),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(8, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[self.cfg.METRIC])

    def train(self, X_train, y_train, X_val, y_val):
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

        class_weights = dict(enumerate(class_weights))
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        ]
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=callbacks, class_weight=class_weights, verbose=1, shuffle=True)

# LSTM:
VOCAB_SIZE = 2000
MAX_LEN = 300 # Token execution window width

class TextPreprocessor:
    def __init__(self, vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')

    def fit(self, texts):
        cleaned = [re.sub(r'[^\w\s]', '', str(t).lower()) for t in texts]
        self.tokenizer.fit_on_texts(cleaned)

    def transform(self, texts):
        cleaned = [re.sub(r'[^\w\s]', '', str(t).lower()) for t in texts]
        seqs = self.tokenizer.texts_to_sequences(cleaned)
        return pad_sequences(seqs, maxlen=self.max_len, padding='post', truncating='pre')

class LSTMTextModel:
    def __init__(self, config, vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
        self.cfg = config
        self.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'lstm_best.keras')

        self.model = keras.Sequential([
            layers.Input(shape=(max_len,)),
            layers.Embedding(input_dim=vocab_size, output_dim=32),
            layers.LSTM(32),
            layers.Dropout(0.6),
            layers.Dense(8, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[self.cfg.METRIC])

    def train(self, X_train, y_train, X_val, y_val):
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights))
        callbacks = [
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
        return self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32, callbacks=callbacks, class_weight=class_weights, verbose=1)

# TRANSFORMER:
BERT_MODEL_NAME = 'distilbert-base-uncased'
BERT_MAX_LEN = 256
BERT_EPOCHS = 3

class DistilBERTClassifier:
    def __init__(self, config, model_name=BERT_MODEL_NAME, max_len=BERT_MAX_LEN):
        self.cfg = config
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
        self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, 'distilbert')
        self.trainer = None

    def _tokenize(self, batch):
        return self.tokenizer(
            batch['text'],
            truncation=True,
            max_length=self.max_len,
            padding='max_length'
        )

    def _build_hf_dataset(self, df):
        cleaned_texts = [str(t).lower() for t in df['lyrics']]
        labels = df['label'].tolist()
        ds = Dataset.from_dict({
            'text': cleaned_texts,
            'labels': labels
        })
        return ds.map(self._tokenize, batched=True, remove_columns=['text'])

    def train(self, train_df, val_df):
        train_ds = self._build_hf_dataset(train_df)
        val_ds = self._build_hf_dataset(val_df)
        collator = DataCollatorWithPadding(self.tokenizer)
        args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            num_train_epochs=BERT_EPOCHS,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            seed=self.cfg.SEED,
            logging_steps=50,
            report_to='none'
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {'accuracy': float(np.mean(preds == labels))}
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        self.trainer.train()

