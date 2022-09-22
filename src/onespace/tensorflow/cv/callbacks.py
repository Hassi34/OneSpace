import tensorflow as tf
import os
from .common import get_unique_filename


def get_callbacks (es_patience, callbacked_model_name ,model_ckpt_path, tensorboard_logs_dir):
    #Tensorboard Callback 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_logs_dir)
    #Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=es_patience, restore_best_weights=True)
    #Model Checkpointing callback (Helpful in backup, would save the last checkpoint in crashing)
    CKPT_name = get_unique_filename(callbacked_model_name, is_model_name=True)
    CKPT_path = os.path.join( model_ckpt_path ,CKPT_name)
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path , save_best_only=True)
    return early_stopping_cb, checkpointing_cb, tensorboard_cb