import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score


def build_model(hp):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=max_length),
        LSTM(hp.Int('units', min_value=64, max_value=256, step=32), return_sequences=True),
        Dropout(0.5),
        LSTM(hp.Int('units', min_value=64, max_value=256, step=32)),
        Dense(1, activation='linear')
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        metrics=['mean_squared_error']
    )
    return model

full_texts = pd.read_csv('train.csv')


# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(full_texts)
train_sequences = tokenizer.texts_to_sequences(full_texts['full_text'])

# Pad sequences
max_length = 500
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')



X_train, X_val, y_train, y_val = train_test_split(train_padded, full_texts['score'], test_size=0.2, random_state=42)


tuner = RandomSearch(
    build_model,
    objective='val_mean_squared_error',
    max_trials=5,
    executions_per_trial=1,
    directory='model_tuning',
    project_name='EssayScoring'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


class MetricsLogger(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_kappas = []

    def on_train_begin(self, logs=None):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_kappas = []
        print("Training started...")

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epochs.append(epoch)
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        print(f"Epoch {epoch + 1} ended. Calculating validation metrics...")

        y_val_pred = self.model.predict(self.validation_data[0])
        y_val_pred_classes = np.round(y_val_pred).astype(int)
        y_val_true_classes = self.validation_data[1].astype(int)
        val_kappa = cohen_kappa_score(y_val_true_classes, y_val_pred_classes, weights='quadratic')

        self.val_kappas.append(val_kappa)

        print(f'Epoch {epoch + 1}: val_kappa: {val_kappa:.4f}')


    def plot_metrics(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(self.epochs, self.val_kappas, label='Validation Kappa')
        plt.title('Validation Kappa')
        plt.xlabel('Epoch')
        plt.ylabel('Kappa')
        plt.legend()

        plt.tight_layout()
        plt.show()

metrics_logger = MetricsLogger(validation_data=(X_val, y_val))

tuner.search(X_train, y_train, epochs=12, validation_data=(X_val, y_val), callbacks=[early_stopping, metrics_logger])

best_model = tuner.get_best_models(num_models=1)[0]

loss, mse = best_model.evaluate(X_val, y_val)

metrics_logger_best = MetricsLogger(validation_data=(X_val, y_val))
best_model.fit(X_train, y_train, epochs=12, validation_data=(X_val, y_val), callbacks=[metrics_logger_best])

loss, mse = best_model.evaluate(X_val, y_val)

metrics_logger_best.plot_metrics()
