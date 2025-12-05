from keras.preprocessing.text import Tokenizer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from keras.preprocessing import sequence
from tensorflow.keras.metrics import AUC
from keras import regularizers
import pandas as pd
import string
import tensorflow_addons as tfa
import numpy as np
from sklearn.model_selection import StratifiedKFold
import numpy as np
import itertools

def load_data():
    df = pd.read_csv("bert_dataset.csv") 
    
    X = df["text"].values
    y = df["generated"].values

    del df
    return X, y

def build_lstm_model():
    vocab_size = 5000
    maxlen = 128
    
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dropout(0.3),  
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),  
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    metrics = [
        tf.metrics.BinaryAccuracy(name='accuracy'),
        AUC(from_logits=False, name='auc'),
        tfa.metrics.F1Score(num_classes=1, threshold=0.5, name='f1', average='macro')
    ]
    
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(1e-4),
        metrics=metrics
    )
    return model

def tf_clean(text, label):
    text = tf.strings.regex_replace(text, rb"[^\x00-\x7F]+", b" ") 
    return text, label


if __name__ == '__main__':
    X, y = load_data()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    metrics_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print("\n" + "="*60)
        print(f"Fold {fold+1}/5")
        print("="*60)
        X_train_raw = X[train_idx]
        y_train_raw = y[train_idx]
    
        X_test_raw = X[test_idx]
        y_test_raw = y[test_idx]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train_raw, y_train_raw)).shuffle(len(X_train_raw)).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_raw, y_test_raw)).batch(32)

        train_ds = train_ds.map(tf_clean)
        test_ds = test_ds.map(tf_clean)
        
        encoder = tf.keras.layers.TextVectorization(max_tokens=5000,output_sequence_length=128)
        encoder.adapt(train_ds.map(lambda text, label: text))
        model = build_lstm_model()

        history = model.fit(
            train_ds,
            validation_data=(test_ds),
            epochs=15,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=5,        
                    restore_best_weights=True,
                    mode='max'
                ),
                TqdmCallback(verbose=1)
            ]
        )

        train_acc = history.history['accuracy'][-1]
        train_auc = history.history['auc'][-1]
        train_f1  = history.history['f1'][-1]
        
        val_acc = history.history['val_accuracy'][-1]
        val_auc = history.history['val_auc'][-1]
        val_f1  = history.history['val_f1'][-1]
        
        save_path = f"./lstm_fold_{fold+1}"
        model.save(save_path)

        metrics_list.append({
            'fold': fold+1,
            'train_accuracy': train_acc,
            'train_auc': train_auc,
            'train_f1': train_f1,
            'val_accuracy': val_acc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'model_path': save_path
        })

        print(f"\nFold {fold+1} Final Results:")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Training AUC: {train_auc:.4f}")
        print(f"Training F1: {train_f1:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Saved model to: {save_path}")

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv('lstm_fold_metrics.csv', index=False)