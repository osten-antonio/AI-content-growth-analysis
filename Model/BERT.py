import pandas as pd
import os
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.metrics import AUC
from official.nlp import optimization  # to create AdamW optimizer
from sklearn.model_selection import StratifiedKFold
from tqdm.keras import TqdmCallback
import tensorflow_addons as tfa


from sklearn.model_selection import train_test_split

print(tf.config.list_physical_devices('GPU'))

# https://stackoverflow.com/questions/44191723/python-library-to-perform-stratified-kfold-cross-validation-in-keras
# https://www.tensorflow.org/text/tutorials/classify_text_with_bert

def load_data():
    df = pd.read_csv('bert_dataset.csv', index_col=False)

    X = df["text"].values        
    y = df["generated"].values    

    return X, y

def build_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1', trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(128, activation='relu', 
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(net)
    net = tf.keras.layers.Dropout(0.3)(net)
    net = tf.keras.layers.Dense(1, activation="sigmoid", name='classifier')(net)
    return tf.keras.Model(text_input, net)
    
def compile_model(model, ds_train):
    steps_per_epoch = tf.data.experimental.cardinality(ds_train).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = [tf.metrics.BinaryAccuracy(),AUC(from_logits=False,name='auc'), tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='macro')]
    
    init_lr = 2e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

    return model


if __name__ == '__main__':
    batch_size = 32
    epochs = 3

    X,y = load_data()
    print(type(y[0]))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print("\n" + "="*60)
        print(f"Fold {fold+1}/5")
        print("="*60)

        X_train_raw = X[train_idx]
        y_train_raw = y[train_idx]

        X_test_raw = X[test_idx]
        y_test_raw = y[test_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train_raw, y_train_raw))
            .shuffle(len(X_train_raw))
            .batch(batch_size)
        )
        test_ds  = tf.data.Dataset.from_tensor_slices((X_test_raw, y_test_raw)).batch(batch_size)

        # Build & train model for this fold

        model = build_model()

        model = compile_model(model,train_ds)
        
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[TqdmCallback(verbose=1)]
        )

        # Save model per fold

        save_path = f"./bert_fold_{fold+1}.h5"
        model.save(save_path)

        print(f"\nFold {fold+1} Final Results:")
        print(f"Training Loss: {history.history['loss'][-1]:.4f}")
        print(f"Training Accuracy: {history.history['binary_accuracy'][-1]:.4f}")
        print(f"Training AUC: {history.history['auc'][-1]:.4f}")
        print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Validation Accuracy: {history.history['val_binary_accuracy'][-1]:.4f}")
        print(f"Validation AUC: {history.history['val_auc'][-1]:.4f}")
        print(f"Training F1: {history.history['f1_score'][-1]:.4f}")
        print(f"Validation F1: {history.history['val_f1_score'][-1]:.4f}")
        print(f"Saved model to: {save_path}")