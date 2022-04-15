import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, metrics
from matplotlib import pyplot as plt
from keras import backend as K
from os.path import exists
from tensorflow.keras import regularizers, initializers
from models import *

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']# used to find to best number of epochs. Set to None to disable
model_names = {}
models_nums = []
model, steps, train = None, None, None
get_model = select_best_models_iteration4


def find_unneeded_columns(d):
    unneeded = []
    dict = {}
    for cat in d[0].columns:
        dict[cat] = []
    for campaign in d:
        for cat in d[0].columns:
            dict[cat].append(campaign[cat].to_numpy().std())
    for cat in d[0].columns:
        if np.array(dict[cat]).std() == 0:
            unneeded.append(cat)
    return unneeded


def train_model(data_sets, epochs, model_number):
    model, model_names = get_model(model_number, max_heats, num_feats)
    X_train, X_val, X_test, y_train, y_val, y_test = data_sets

    model_name = model_names[model_number]
    if load == "y" or load == "yes":
        if exists(f'weights/{model_name}_model_weights.h5'):
            model.load_weights(f'weights/{model_name}_model_weights.h5')
        else:
            print("model weights do not exist! Training new model ...")

    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

    model.save_weights(f'weights/{model_name}_model_weights.h5')

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("test_loss:", test_loss, "test_acc", test_acc)
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
    print("train_loss:", train_loss, "train_acc", train_acc)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=2)
    print("val_loss:", val_loss, "val_acc", val_acc)
    predictions = model.predict(data)
    return history, model, predictions, test_acc, train_acc, val_acc


def calculate_RMSE_whole_set(predictions):
    rmse_whole_set = np.sqrt(np.mean(np.square(predictions.reshape(-1) - targets)))
    print("RMSE Whole Set:", rmse_whole_set)
    return rmse_whole_set


def show_RMSE(history, c, model_name, show=False, type="both"):
    c = c % 7
    color = colors[c]
    if type == "both":
        plt.plot(history.history['root_mean_squared_error'], label=f'{model_name} - RMSE', color='r')
        plt.plot(history.history['val_root_mean_squared_error'], label=f'{model_name} - Validation-RMSE', color='g')
        plt.title(f'RMSE over Epochs')
    elif type == "val":
        plt.plot(history.history['val_root_mean_squared_error'], label=f'{model_name}', color=color)
        plt.title(f'RMSE of the Validation Set over Epochs')
    elif type == "train":
        plt.plot(history.history['root_mean_squared_error'], label=f'{model_name}', color=color)
        plt.title(f'RMSE of the Training Set over Epochs')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.ylim(0, 30)
    plt.legend()
    if show:
        plt.show()


def show_barchart(predictions, model_name, rmse):
    n_groups = len(predictions)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, predictions.reshape(-1), bar_width, alpha=opacity, color='b', label='Predictions')
    plt.bar(index + bar_width, targets, bar_width, alpha=opacity, color='g', label='Targets')

    plt.xlabel('Campaigns')
    plt.ylabel('Wear [mm]')
    plt.title(f'Wear Predictions - {model_name}\nRMSE: {rmse:.2f} ')
    plt.legend()
    plt.show()


def show_residuals(predictions, model_name, rmse):
    n_groups = len(predictions)
    index = np.arange(n_groups)
    plt.bar(index, predictions.reshape(-1) - targets, label='Residuals')
    plt.xlabel('Campaigns')
    plt.ylabel('Residuals [mm]')
    plt.title(f'Residuals between Predictions and Targets - {model_name}\nRMSE: {rmse:.2f} ')
    plt.legend()
    plt.show()


def train_single_model(model_number):
    global model, model_names
    model, model_names = get_model(model_number, max_heats, num_feats)
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    data_sets = [X_train, X_val, X_test, y_train, y_val, y_test]
    history, model, predictions, _, _, _ = train_model(data_sets, epochs, model_number)
    rmse = calculate_RMSE_whole_set(predictions)
    show_RMSE(history, model_number,  model_names[model_number])
    show_barchart(predictions, model_names[model_number], rmse)
    show_residuals(predictions, model_names[model_number], rmse)


def train_and_compare_models():
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    data_sets = [X_train, X_val, X_test, y_train, y_val, y_test]

    results = {}
    for model_number in models_nums:
        history, model, predictions, test_acc, train_acc, val_acc = train_model(data_sets, epochs, model_number)
        rmse_whole = calculate_RMSE_whole_set(predictions)
        results[model_number] = {'h': history, 'number': model_number, 'rmse_train': train_acc,
                                 'rmse_val': val_acc, 'rmse_test': test_acc, 'rmse_whole': rmse_whole}
    for model_number in models_nums:
        r = results[model_number]
        show_RMSE(r['h'], r['number'], model_names[model_number], show=False, type="val")
    plt.show()
    for model_number in models_nums:
        r = results[model_number]
        show_RMSE(r['h'], r['number'], model_names[model_number], show=False, type="train")
    plt.show()

    fig, ax = plt.subplots()
    index = np.arange(len([model_names[results[r]['number']] for r in results]))
    bar_width = 0.15
    ax.bar(index, [results[r]['rmse_train'] for r in results], bar_width, color='b', label='Train Set')
    ax.bar(index + bar_width, [results[r]['rmse_val'] for r in results], bar_width, color='c', label='Validation Set')
    ax.bar(index + bar_width * 2, [results[r]['rmse_test'] for r in results], bar_width, color='g', label='Test Set')
    ax.bar(index + bar_width * 3, [results[r]['rmse_whole'] for r in results], bar_width, color='y', label='Entire Set')
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Model RMSE of Test Set & the entire set')
    plt.xticks(index + bar_width * 1.5, [model_names[results[r]['number']] for r in results])
    plt.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
    print(results)


def show_and_compare_training_progression():
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    data_sets = [X_train, X_val, X_test, y_train, y_val, y_test]
    results = {}
    global load
    load = "yes"
    for model_number in models_nums:
        cur_step = 0
        train = []
        val = []
        test = []
        whole = []
        x = []
        while True:
            history, model, predictions, test_acc, train_acc, val_acc = train_model(data_sets, steps, model_number)
            cur_step += steps
            x.append(cur_step)
            whole.append(calculate_RMSE_whole_set(predictions))
            test.append(test_acc)
            val.append(val_acc)
            train.append(train_acc)
            if cur_step >= epochs:
                results[model_number] = {'number': model_number, 'x': x, 'whole': whole, 'test': test,
                                         'train': train, 'val': val}
                break

    for x in ['train', 'val', 'test', 'whole']:
        for model_number in models_nums:
            result = results[model_number]
            c = model_number % 7
            color = colors[c]
            model_name = model_names[model_number]
            plt.plot(result['x'], result[x], label=f'{model_name}', color=color)
        plt.title(f'RMSE of the {x} set over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.show()

    print(results)


# load data
with open('data/dataset_jsis_ladle_wall_metalzone_cut_prep_20220329_105019.pkl', 'rb') as f:
    [list_prod_prepared, HS_targets_prepared, list_abstellgrund_prepared, list_initial] = pickle.load(f)
    f.close()

# remove empty and unneeded columns
data = []
unneeded_columns = find_unneeded_columns(list_prod_prepared)
for campaign in list_prod_prepared:
    unneeded_columns.extend(['Heat ID', 'Campaign'])
    c = campaign.drop(unneeded_columns, axis=1)
    data.append(np.array(c.values.tolist()))

# get dimensions
max_heats = max(d.shape[0] for d in data)
num_feats = data[0].shape[1]

# select models
_, model_names = get_model(None, max_heats, num_feats)
models_nums = list(model_names.keys())

# pad every item to the same number of heats
for i, d in enumerate(data):
    if max_heats != d.shape[0]:
        data[i] = np.append(d, np.array([d[-1]] * (max_heats - d.shape[0])), axis=0)

# reshape data
data, targets = np.array(data).reshape((-1, max_heats, num_feats, 1)), np.array(HS_targets_prepared)

# read what to do from the CL
model_number = input("What model to you want to use:\n"
                     + "\n".join([f'Enter "{x}" to use the {model_names[x]}' for x in models_nums]) +
                     "\nEnter 'c' to compare all models"
                     "\nEnter 'b' to evaluate the best model\n").strip().strip("'").strip('"')
model_number = int(model_number) if model_number.isnumeric() else model_number
if model_number in ['b', 'B']:
    model, model_names = get_best_model(1, max_heats, num_feats)
    model_name = model_names[1]
    if exists(f'best_weights/{model_name}_model_weights.h5'):
        model.load_weights(f'best_weights/{model_name}_model_weights.h5')
        predictions = model.predict(data)
        rmse = calculate_RMSE_whole_set(predictions)
        show_barchart(predictions, model_name, rmse)
        show_residuals(predictions, model_name, rmse)
    exit(0)

if model_number in ['c', 'C']:
    steps = input("If you want to compare the results after a certain amount of epochs, press Enter\n"
                  "If you want to periodically compare all models while they are trained, enter the step size:\n")
    steps = int(steps) if steps.isnumeric() else None
if model_number not in models_nums and model_number not in ['c', 'C']:
    model_number = 1

load = input("Load existing model weights? (enter 'y' or 'n'):\n").strip().strip("'").strip('"').lower()

if load in ["y", "yes"]:
    train = input("Do you want to continue training this model (enter 't') or evaluate its current state? (enter anything else):\n"
                  "").strip().strip("'").strip('"')
if train in [None, "t", "train"]:
    epochs = input("Enter the number of epochs to train (default = 250):\n").strip().strip("'").strip('"')
    epochs = 250 if not epochs.isnumeric() else int(epochs)

# compare models
if model_number == "c" or model_number == "C":
    if steps is None:
        train_and_compare_models()
    else:
        show_and_compare_training_progression()
else:
    if train in ["t","train"]:
        train_single_model(model_number)
    else:
        model, model_names = get_model(model_number, max_heats, num_feats)
        model_name = model_names[model_number]
        if exists(f'weights/{model_name}_model_weights.h5'):
            model.load_weights(f'weights/{model_name}_model_weights.h5')
            predictions = model.predict(data)
            rmse = calculate_RMSE_whole_set(predictions)
            show_barchart(predictions, model_names[model_number], rmse)
            show_residuals(predictions, model_names[model_number], rmse)
        else:
            print("model weights do not exist!")

