# models.py

import tensorflow as tf

def load_xray_model():
    model = tf.keras.models.load_model('models/xray_model.h5')
    return model

def load_ecg_model():
    model = tf.keras.models.load_model('models/ecg_model.h5')
    return model

# Add similar functions for ultrasound, CT scan, MRI models
