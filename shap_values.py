from important_models import *
import numpy as np
from tensorflow.keras import layers, models, metrics
from os.path import exists
import random
from scipy.stats import pearsonr
import shap
import preprocessing
from sklearn.model_selection import train_test_split
from preprocessing import *
import tensorflow as tf


def get_shap_values(get_model, model_number, dataset, normalize, random_state):

    if isinstance(dataset, str):
        data, targets, max_heats, num_feats, original_lengths, column_names = preprocessing.prepare_data(dataset, all=True) #'data/dataset_jsis_ladle_wall_metalzone_cut_prep_20220329_105019.pkl'
    else:
        data, targets, max_heats, num_feats, original_lengths, column_names = dataset
    targets = targets / original_lengths * max_heats if normalize else targets
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=random_state)
    model, model_names, multi_input = get_model(model_number, max_heats, num_feats)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)

    model.fit(X_train, y_train, epochs=50, callbacks=[callback])

    explainer = shap.DeepExplainer(model = model, data = X_train)
    shap_values = explainer.shap_values(X_test)

    shap_values_reshaped = []
    for i in range(len(X_test)):
        shap_values_reshaped.append(np.mean(np.abs(shap_values[0][i].reshape(max_heats,num_feats)), axis=0))
    shap_values_reshaped = np.array(shap_values_reshaped)

    max_values = np.mean(np.array(shap_values_reshaped), axis=0)
    max_indexes = np.argsort(-max_values)
    print(np.array(column_names)[max_indexes])

    return shap_values, shap_values_reshaped, X_test, column_names, max_indexes

