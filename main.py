import tensorflow as tf
import keras.backend as K

import random

from visualization import *
from preprocessing import *
from important_models import *

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from os.path import exists
from scipy.stats import pearsonr
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime
from csv import writer
from tensorflow.keras.utils import plot_model

# HYPERPARAMETERS

get_model = final_model_ms_sl_dataset  # which models to use.
#                            Look at the important_models.py file to see some tested collections of models
random_state = 1  # Choose a number to recreate the split of training, validation test sets or set it to None.
#                   This is not used when using leave one out cross validaiton

use_all_datasets = False  # Execute the selected function for all selected datasets
selected_single_dataset = 'Metalzone'  # If use_all_datasets is False, a single data set is used.
#                            Choose one of 'Metalzone', 'Slagzone', 'Bottom Inlet', 'Bottom Outlet', 'Heart'
selected_dataset = 'mzsl'  # If use_all_datasets is True, all of a certain set are used.
#                            selected_dataset = 'mzsl' --> Metalzone, Slagzone
#                            selected_dataset = 'voest' --> Bottom Inlet, Bottom Outlet, Heart
#                            selected_dataset = 'all' --> Metalzone, Slagzone, Bottom Inlet, Bottom Outlet, Heart

cross_validation = True  # should cross validation be done or should only one model be trained
num_fold_cv_init = 5  # number folds in the cross validation. 'leave_one_out' enabled leave-one-out-CV
zero_padding = True  # if this is false, the last value is repeated util every campaign is the same length

normalize_campaign_size = False  # should the targets be divided by the length of the corresponding campaign
data_aug = False  # Use mix-up data augmentation to create more data for the training and validation set
data_aug_shuffle = False  # Shuffle heats in a campaign to create new ones. Apply uniform noise to targets of new ones.
aug_amount = 50  # how much new data is created during data augmentation. 1 = size of dataset

loss_function = root_mean_squared_error  # which loss function is used to train a model.
#                                       Choose 'root_mean_squared_error' or 'focal_loss'
focal_loss_gamma = 0.2  # This is only used if the focal loss is used
init_loss_function(loss_function, focal_loss_gamma)

gauss_filter = True  # should a 1D Gauss Filter be applied to fitting features
normalize_data = False  # should the data be normalized between 0 and 1 (data is already provided normalized)
early_stopping = True  # should early stopping be used during training

combine_datasets = False  # Training a single model for all three of the hard datasets
classification = False  # Training a classification model instead of a regression model

use_columns = 5  # which features are used. Use 'all', 'best' for pre-selected one
#                  or a numeric value to use the features with the highest shap-values

plot_shap_differences = False  # iterate over a range (2-25) of the selected number of features to compare the results
plot_model_architecture = True  # print model.summary()

# initialization of some variables
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
model_names = {}
models_nums = []
model, steps, train = None, None, None
multi_input = False
cols = []
random.seed(random_state)


# As a sanity check the results can be compared to the average target
def use_mean_targets_cross_validation(xx, yy):
    targets = yy

    if normalize_campaign_size:
        yy = np.array(yy) / original_lengths * max_heats

    mean = np.mean(yy)
    preds = [mean] * len(xx)
    preds = np.array(preds) / max_heats * original_lengths if normalize_campaign_size else np.array(preds)
    rmse_whole = np.sqrt(np.mean(np.square(preds - np.array(targets))))
    print("mean_target - RMSE if fitted on whole set", rmse_whole)

    sets_x = np.array_split(targets, num_fold_cv)
    sets_ol = np.array_split(original_lengths, num_fold_cv)
    predictions, test_accs = [], []
    for test_index in range(num_fold_cv):
        x, ol = sets_x, sets_ol
        X_test, ol_test = x[test_index], ol[test_index]
        x = np.delete(x, (test_index))
        ol = np.delete(ol, (test_index))
        if len(sets_x) != num_fold_cv:
            X_train = np.concatenate(x)
            ol_train = np.concatenate(ol)
        else:
            X_train = x
            ol_train = ol

        if normalize_campaign_size:
            X_train = np.array(X_train) / ol_train * max_heats

        if data_aug:
            X_train, ol_train, _ = data_augmentation(X_train, ol_train, lengths=ol_train)

        if not isinstance(X_train[0], float):
            X_train = [c[0] for c in X_train]
        mean_train = np.mean(X_train)
        preds = [mean_train] * len(X_test)
        preds = np.array(preds) / max_heats * np.array(ol_test) if normalize_campaign_size else np.array(preds)
        predictions.append(preds)
        rmse_test = np.sqrt(np.mean(np.square(preds - np.array(X_test))))
        test_accs.append(rmse_test)
        print("test_rmse", test_index, rmse_test)

    predictions = np.concatenate(predictions)
    rmse_whole = calculate_RMSE_whole_set(predictions)
    print("rmse_whole", rmse_whole)
    average_test_rmse = sum(test_accs) / num_fold_cv
    print("average_test_rmse", average_test_rmse)
    show_barchart(predictions, "Mean Target", rmse_whole, t)
    show_residuals(predictions, "Mean Target", rmse_whole, t)
    return rmse_whole


# linear regression between some feature and the targets
def linear_regression_cross_validation(xx, yy):
    targets = yy
    if normalize_campaign_size:
        ol_train = np.array(xx)
        yy = np.array(yy) / ol_train * max_heats

    slope, intercept, r, p, std_err = stats.linregress(xx, yy)

    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, xx))

    preds = []
    for i in xx:
        preds.append(myfunc(i))
    preds = np.array(preds) / max_heats * np.array(xx) if normalize_campaign_size else np.array(preds)
    rmse_whole = np.sqrt(np.mean(np.square(preds - np.array(targets))))
    print("linear_regression - RMSE if fitted on whole set", rmse_whole)

    plt.scatter(xx, yy)
    plt.plot(xx, mymodel)
    corr, _ = pearsonr(xx, yy)
    plt.xlabel("Campaign Lengths")
    plt.ylabel("Scaled Targets")
    plt.title('Whole Set - Pearsons Correlation %.3f' % corr)
    plt.show()

    sets_x = np.array_split(xx, num_fold_cv)
    sets_y = np.array_split(targets, num_fold_cv)

    predictions, test_accs = [], []
    for test_index in range(num_fold_cv):
        x, y = sets_x, sets_y
        X_test, y_test = x[test_index], y[test_index]
        x = np.delete(x, (test_index))
        y = np.delete(y, (test_index))

        if not isinstance(x.tolist()[0], int):
            X_train = np.concatenate(x)
            y_train = np.concatenate(y)
        else:
            X_train = x
            y_train = y

        if normalize_campaign_size:
            y_train = np.array(y_train) / np.array(X_train) * max_heats

        if data_aug:
            X_train, y_train, _ = data_augmentation(X_train, y_train, lengths=y_train)

        slope, intercept, r, p, std_err = stats.linregress(X_train, y_train)

        def myfunc_test(x):
            return slope * x + intercept

        preds = []
        for i in X_test:
            preds.append(myfunc_test(i))

        preds = np.array(preds) / max_heats * np.array(X_test) if normalize_campaign_size else np.array(preds)
        predictions.append(preds)
        rmse_test = np.sqrt(np.mean(np.square(preds - np.array(y_test))))
        test_accs.append(rmse_test)
        print("test_rmse", test_index, rmse_test)

    predictions = np.concatenate(predictions)
    rmse_whole = calculate_RMSE_whole_set(predictions)
    print("rmse_whole", rmse_whole)
    average_test_rmse = sum(test_accs) / num_fold_cv
    print("average_test_rmse", average_test_rmse)
    show_barchart(predictions, "Linear Regression", rmse_whole, t)
    show_residuals(predictions, "Linear Regression", rmse_whole, t)
    return rmse_whole


# calculating the RMSE of the whole dataset
def calculate_RMSE_whole_set(predictions, tar=None):
    global d, t
    rmse_whole_set = np.sqrt(np.mean(np.square(predictions.reshape(-1) - (t if tar is None else tar))))
    print("RMSE Whole Set:", rmse_whole_set)
    return rmse_whole_set


# training and evaluating a single model
def train_single_model(model_number, ffcv, data, targets):
    global model, model_names, d, t, load, original_lengths

    if ffcv:
        load = "no"

        sets_x = np.array_split(data, num_fold_cv)
        sets_y = np.array_split(targets, num_fold_cv)
        sets_ol = np.array_split(original_lengths, num_fold_cv)

        preds, test_accs = [], []
        for test_index in range(num_fold_cv):
            x, y, lengths = sets_x, sets_y, sets_ol
            X_test, y_test, ol_test = x[test_index], y[test_index], lengths[test_index]
            x = np.delete(x, (test_index), axis=0)
            y = np.delete(y, (test_index), axis=0)
            lengths = np.delete(lengths, (test_index), axis=0)
            X_train = np.concatenate(x)
            y_train = np.concatenate(y)
            ol_train = np.concatenate(lengths)
            data_sets = [X_train, X_test, y_train, y_test, ol_train, ol_test]

            _, model, predictions, test_preds, test_rmse, train_rmse, whole_rmse = \
                train_model(data_sets, epochs, model_number, load=load)
            preds.append(test_preds)
            test_accs.append(test_rmse.numpy() if not combine_datasets else test_rmse)
        if combine_datasets:
            for i in range(3):

                predictions_ = np.array([x[i][0] for x in preds])
                targets_ = np.array([x[i] for x in t])

                if np.isnan(predictions).any():

                    targets_ = np.array([x for i, x in enumerate(targets_) if i not in np.argwhere(np.isnan(predictions_))[:, 0]])

                    predictions_ = np.array([x for i, x in enumerate(predictions_) if i not in
                                             np.argwhere(np.isnan(predictions_))[:, 0]])

                else:
                    predictions = np.array([x[i][0] for x in preds])
                    targets_ = np.array([x[i] for x in t])

                rmse = np.sqrt(np.mean(np.square(predictions_ - targets_)))

                print("Test-set-predictions-RMSE", rmse)
                show_barchart(predictions.reshape(-1), model_names[model_number],
                              f"Average-Test-Set-RMSE: {rmse[:, i]:.2f}", t[:, i])
                show_residuals(predictions.reshape(-1), model_names[model_number],
                               f"Average-Test-Set-RMSE: {rmse[:, i]:.2f}", t[:, i])
        else:
            predictions = np.concatenate(preds)
            print("Test-RMSE-average:", sum(test_accs) / num_fold_cv)
            rmse = calculate_RMSE_whole_set(predictions)
            print("Test-set-predictions-RMSE", rmse)
            show_barchart(predictions.reshape(-1), model_names[model_number],
                          f"Average-Test-Set-RMSE: {rmse:.2f}", t)
            show_residuals(predictions.reshape(-1), model_names[model_number],
                           f"Average-Test-Set-RMSE: {rmse:.2f}", t)
    else:
        model, model_names, multi_input = get_model(model_number, max_heats, num_feats)
        X_train, X_test, y_train, y_test, ol_train, ol_test = \
            train_test_split(data, targets, original_lengths, test_size=0.2, random_state=random_state)
        data_sets = [X_train, X_test, y_train, y_test, ol_train, ol_test]
        history, model, predictions, _, test_rmse, train_rmse, whole_rmse = \
            train_model(data_sets, epochs, model_number, load=load)

        show_RMSE_over_epochs(history, model_number, model_names[model_number])
        show_barchart(predictions, model_names[model_number],
                      f"Test-Set-RMSE: {test_rmse:.2f}, Whole-Set-RMSE: {whole_rmse:.2f}", t)
        show_residuals(predictions, model_names[model_number],
                       f"Test-Set-RMSE: {test_rmse:.2f}, Whole-Set-RMSE: {whole_rmse:.2f}", t)


# training multiple models and comparing the results
def train_and_compare_models(data, targets, file=""):
    global d, t
    results = {}
    if combine_datasets:
        final_results = [[file_titles[0]], [file_titles[1]], [file_titles[2]]]
    else:
        mean_target = use_mean_targets_cross_validation(data, targets)
        lr = linear_regression_cross_validation(original_lengths, targets)
        final_results = [file, mean_target, lr]

    if cross_validation:
        start = datetime.now()
        sets_x = np.array_split(data, num_fold_cv)
        sets_y = np.array_split(targets, num_fold_cv)
        sets_ol = np.array_split(original_lengths, num_fold_cv)
        for model_number in models_nums:
            preds, test_accs, train_accs = [], [], []
            for test_index in range(num_fold_cv):

                print(f"{file} - Training model #{model_number}! Iteration {(test_index + 1)} of {num_fold_cv} ...")
                x, y, lengths = sets_x, sets_y, sets_ol
                X_test, y_test, ol_test = x[test_index], y[test_index], lengths[test_index]

                if len(sets_x) != num_fold_cv:
                    x = np.delete(x, (test_index))
                    y = np.delete(y, (test_index))
                    lengths = np.delete(lengths, (test_index))
                else:
                    x = np.delete(x, (test_index), axis=0)
                    y = np.delete(y, (test_index), axis=0)
                    lengths = np.delete(lengths, (test_index), axis=0)

                X_train = np.concatenate(x)
                y_train = np.concatenate(y)
                ol_train = np.concatenate(lengths)

                data_sets = [X_train, X_test, y_train, y_test, ol_train, ol_test]
                _, model, predictions, test_preds, test_rmse, train_rmse, whole_rmse = \
                    train_model(data_sets, epochs, model_number, load="no")
                preds.append(test_preds)
                test_accs.append(test_rmse.numpy() if not combine_datasets else test_rmse)
                train_accs.append(train_rmse.numpy() if not combine_datasets else train_rmse)
            if combine_datasets:
                for i in range(3):
                    print(file_titles[i])
                    preds = np.array(preds)
                    predictions = np.concatenate(preds[:, i])
                    test_rmse = sum(np.array(test_accs)[:, i]) / num_fold_cv
                    train_rmse = sum(np.array(train_accs)[:, i]) / num_fold_cv
                    print("Test-RMSE-average:", test_rmse)
                    whole_rmse = calculate_RMSE_whole_set(predictions, tar=targets[:, i])
                    print("Test-set-predictions-RMSE", whole_rmse)
                    final_results[i].append(whole_rmse)
                    show_barchart(predictions, model_names[model_number],
                                  f"{file_titles[i]} - Average-Test-Set-RMSE: {whole_rmse:.2f}", targets[:, i])
                    show_residuals(predictions, model_names[model_number],
                                   f"{file_titles[i]} - Average-Test-Set-RMSE: {whole_rmse:.2f}", targets[:, i])
                    results[model_number] = {'h': None, 'number': model_number, 'rmse_train': train_rmse,
                                             'rmse_test': test_rmse, 'rmse_whole': whole_rmse}
            else:
                predictions = np.concatenate(preds)
                if np.isnan(predictions).any():
                    whole_test_rmse = np.sqrt(np.mean(
                        np.square(np.array([x for i, x in enumerate(predictions) if i not in
                                            np.argwhere(np.isnan(predictions))[:, 0]]) - np.array(
                        [x for i, x in enumerate(t) if i not in np.argwhere(np.isnan(predictions))[:, 0]]))))
                else:
                    whole_test_rmse = calculate_RMSE_whole_set(predictions)
                test_rmse = sum(test_accs) / num_fold_cv
                train_rmse = sum(train_accs) / num_fold_cv
                print("Test-RMSE-average:", test_rmse)
                print("Test-set-predictions-RMSE", whole_test_rmse)
                final_results.append(whole_test_rmse)
                show_barchart(predictions, model_names[model_number],
                              f"{file} - Average-Test-Set-RMSE: {whole_test_rmse:.2f}", t)
                show_residuals(predictions, model_names[model_number],
                               f"{file} - Average-Test-Set-RMSE: {whole_test_rmse:.2f}", t)
                results[model_number] = {'h': None, 'number': model_number, 'rmse_train': train_rmse,
                                         'rmse_test': test_rmse, 'rmse_whole': whole_test_rmse}
        end = datetime.now()
        print("Elapsed time:", end - start)
    else:
        X_train, X_test, y_train, y_test, ol_train, ol_test = \
            train_test_split(data, targets, original_lengths, test_size=0.2, random_state=random_state)
        data_sets = [X_train, X_test, y_train, y_test, ol_train, ol_test]

        for model_number in models_nums:
            print(f"Training model #{model_number} ...")
            history, model, predictions, _, test_rmse, train_rmse, whole_rmse = \
                train_model(data_sets, epochs, model_number, load="no")
            final_results.append(test_rmse)
            results[model_number] = {'h': history, 'number': model_number, 'rmse_train': train_rmse.numpy(),
                                     'rmse_test': test_rmse.numpy(),
                                     'rmse_whole': whole_rmse.numpy()}
        for model_number in models_nums:
            r = results[model_number]
            show_RMSE_over_epochs(r['h'], r['number'], model_names[model_number], show=False, type="train")
        plt.show()

    fig, ax = plt.subplots()
    index = np.arange(len([model_names[results[r]['number']] for r in results]))
    bar_width = 0.33
    ax.bar(index, [results[r]['rmse_train'] for r in results], bar_width, color='b', label='Train Set')
    ax.bar(index + bar_width * 1, [results[r]['rmse_whole'] for r in results], bar_width, color='g', label='Test Set')
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title(f'{file} - Model RMSE of Test Set & the entire set')
    plt.xticks(index + bar_width * 0.33, [model_names[results[r]['number']] for r in results])
    plt.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

    with open('results.csv', 'a', newline='') as f_object:
        writer_object = writer(f_object)
        if combine_datasets:
            writer_object.writerow(final_results[0])
            writer_object.writerow(final_results[1])
            writer_object.writerow(final_results[2])
        else:
            writer_object.writerow(final_results)
        f_object.close()


# trains multiple models for mutliple short steps. Afterwards the results are evaluated and compared.
def show_and_compare_training_progression(data, targets):
    X_train, X_test, y_train, y_test, ol_train, ol_test = \
        train_test_split(data, targets, original_lengths, test_size=0.2, random_state=random_state)
    data_sets = [X_train, X_test, y_train, y_test, ol_train, ol_test]

    results = {}
    global load, d, t
    load = "yes"
    for model_number in models_nums:
        cur_step, train, test, whole, x = 0, [], [], [], []
        while True:
            history, model, predictions, _, test_rmse, train_rmse, whole_rmse = \
                train_model(data_sets, steps, model_number, load=load)
            cur_step += steps
            x.append(cur_step)
            whole.append(whole_rmse)
            test.append(test_rmse)
            train.append(train_rmse)
            if cur_step >= epochs:
                results[model_number] = {'number': model_number, 'x': x, 'whole': whole, 'test': test,
                                         'train': train}
                show_RMSE_over_epochs(history, model_number, model_names[model_number])
                show_barchart(predictions, model_names[model_number],
                              f"Test-Set-RMSE: {test_rmse:.2f}, Whole-Set-RMSE: {whole_rmse:.2f}", t)
                show_residuals(predictions, model_names[model_number],
                               f"Test-Set-RMSE: {test_rmse:.2f}, Whole-Set-RMSE: {whole_rmse:.2f}", t)
                break

    for x in ['train', 'test', 'whole']:
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


# regardless which parameters have been chosen, this function prepares the input data, trains the model
# and evaluates the results
def train_model(data_sets, epochs, model_number, load="no"):
    global d, t, original_lengths
    model, model_names, multi_input = get_model(model_number, max_heats, num_feats,
                                                outputs=3 if combine_datasets else 1)
    X_train, X_test, y_train, y_test, ol_train, ol_test = data_sets

    model_name = model_names[model_number]

    model.summary()
    if load == "y" or load == "yes":
        if exists(f'weights/{model_name}_model_weights.h5'):
            model.load_weights(f'weights/{model_name}_model_weights.h5')
        else:
            print("model weights do not exist! Training new model ...")

    if normalize_campaign_size:
        yy = y_train / ol_train * max_heats
    else:
        yy = y_train

    if data_aug:
        xx, yy, ol_train_aug = data_augmentation(X_train, yy, lengths=ol_train)
    else:
        xx, yy, ol_train_aug = X_train, yy, ol_train


    if data_aug_shuffle:
        xx, yy, ol_train_aug = shuffle_data_augmentation(X_train, yy, lengths=ol_train)
        if gauss_filter:
            xx = apply_gauss_filter(xx)
            xx = np.array([x[0].reshape(x[0].shape + (1,)) for x in xx])
    else:
        xx, yy, ol_train_aug = X_train, yy, ol_train


    if multi_input:
        X_model_input, Y_model_input = (xx, np.array(ol_train_aug).reshape(-1, 1)), yy
        X_model_input_test, Y_model_input_test = (X_test, np.array(ol_test).reshape(-1, 1)), y_test
    elif multi_input is None:
        X_model_input, X_model_input_test = [], []
        for i in range(num_feats):
            X_model_input.append(xx[:, :, i])
            X_model_input_test.append(X_test[:, :, i])
        Y_model_input = yy
        Y_model_input_test = y_test
    else:

        X_model_input, Y_model_input = xx, yy
        X_model_input_test, Y_model_input_test = X_test, y_test

    if classification:
        Y_model_input, Y_model_input_test = np.array(Y_model_input), np.array(Y_model_input_test)

    if early_stopping:
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        history = model.fit(X_model_input, Y_model_input, epochs=epochs, callbacks=[callback])
    else:
        history = model.fit(X_model_input, Y_model_input, epochs=epochs)

    model.save_weights(f'weights/{model_name}_model_weights.h5')

    if classification:
        test_prediction = model.predict(X_model_input_test)
        test_acc = accuracy_score(Y_model_input_test.argmax(axis=1), test_prediction.argmax(axis=1))
        cm = confusion_matrix(Y_model_input_test.argmax(axis=1), test_prediction.argmax(axis=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Test Accuracy {test_acc:.2}")
        plt.show()

        train_prediction = model.predict(X_model_input)
        train_acc = accuracy_score(Y_model_input.argmax(axis=1), train_prediction.argmax(axis=1))
        cm = confusion_matrix(Y_model_input.argmax(axis=1), train_prediction.argmax(axis=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Train Accuracy {train_acc:.2}")
        plt.show()

        whole_prediction = model.predict(d)
        whole_acc = accuracy_score(t.argmax(axis=1), whole_prediction.argmax(axis=1))
        cm = confusion_matrix(t.argmax(axis=1), whole_prediction.argmax(axis=1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"Whole Set Accuracy {whole_acc:.2}")
        plt.show()
    elif combine_datasets:
        test_preds_list, test_rmse_list, train_rmse_list, whole_rmse_list = [], [], [], []
        test_prediction = model.predict(X_model_input_test)[:, 0]
        test_preds = test_prediction / max_heats * ol_test if normalize_campaign_size else test_prediction
        test_rmse = root_mean_squared_error(y_test[:, 0], test_preds)
        test_preds_list.append(test_preds)
        test_rmse_list.append(test_rmse)
        print(f"RMSE Test-Accuracy Model 1: {test_rmse}")
        test_prediction = model.predict(X_model_input_test)[:, 1]
        test_preds = test_prediction / max_heats * ol_test if normalize_campaign_size else test_prediction
        test_rmse = root_mean_squared_error(y_test[:, 1], test_preds)
        test_preds_list.append(test_preds)
        test_rmse_list.append(test_rmse)
        print(f"RMSE Test-Accuracy Model 2: {test_rmse}")
        test_prediction = model.predict(X_model_input_test)[:, 2]
        test_preds = test_prediction / max_heats * ol_test if normalize_campaign_size else test_prediction
        test_rmse = root_mean_squared_error(y_test[:, 2], test_preds)
        test_preds_list.append(test_preds)
        test_rmse_list.append(test_rmse)
        print(f"RMSE Test-Accuracy Model 3: {test_rmse}")
        test_prediction = model.predict(X_model_input_test).reshape(-1, 1)
        test_preds = test_prediction / max_heats * ol_test if normalize_campaign_size else test_prediction
        test_rmse = root_mean_squared_error(y_test.reshape(-1, 1), test_preds)
        print(f"RMSE Test-Accuracy Model Average: {test_rmse}")

        train_prediction = model.predict(
            (X_train, np.array(ol_train).reshape(-1, 1))) if multi_input else model.predict(X_train)
        train_prediction = train_prediction[:, 0].reshape(-1)
        train_preds = train_prediction / max_heats * ol_train if normalize_campaign_size else train_prediction
        train_rmse = root_mean_squared_error(y_train[:, 0], train_preds.reshape(-1))
        train_rmse_list.append(train_rmse)
        print(f"RMSE Training-Accuracy Model 1: {train_rmse}")
        train_prediction = model.predict(
            (X_train, np.array(ol_train).reshape(-1, 1))) if multi_input else model.predict(X_train)
        train_prediction = train_prediction[:, 1].reshape(-1)
        train_preds = train_prediction / max_heats * ol_train if normalize_campaign_size else train_prediction
        train_rmse = root_mean_squared_error(y_train[:, 1], train_preds.reshape(-1))
        train_rmse_list.append(train_rmse)
        print(f"RMSE Training-Accuracy Model 2: {train_rmse}")
        train_prediction = model.predict(
            (X_train, np.array(ol_train).reshape(-1, 1))) if multi_input else model.predict(X_train)
        train_prediction = train_prediction[:, 2].reshape(-1)
        train_preds = train_prediction / max_heats * ol_train if normalize_campaign_size else train_prediction
        train_rmse = root_mean_squared_error(y_train[:, 2], train_preds.reshape(-1))
        train_rmse_list.append(train_rmse)
        print(f"RMSE Training-Accuracy Model 3: {train_rmse}")
        train_prediction = model.predict(
            (X_train, np.array(ol_train).reshape(-1, 1))) if multi_input else model.predict(X_train)
        train_prediction = train_prediction.reshape(-1, 1)
        train_preds = train_prediction / max_heats * ol_train if normalize_campaign_size else train_prediction
        train_rmse = root_mean_squared_error(y_train.reshape(-1), train_preds.reshape(-1))
        print(f"RMSE Training-Accuracy Model Average: {train_rmse}")

        whole_prediction = model.predict(
            (np.array(d), np.array(original_lengths).reshape(-1, 1))) if multi_input else model.predict(np.array(d))
        whole_preds = whole_prediction / max_heats * original_lengths if normalize_campaign_size else whole_prediction
        whole_rmse = root_mean_squared_error(t.reshape(-1), whole_preds.reshape(-1))
        print(f"RMSE Whole Dataset: {whole_rmse}")
        return history, model, whole_preds, test_preds_list, test_rmse_list, train_rmse_list, whole_rmse
    else:
        test_prediction = model.predict(X_model_input_test).reshape(-1)
        test_preds = test_prediction / max_heats * ol_test if normalize_campaign_size else test_prediction
        test_rmse = root_mean_squared_error(y_test, test_preds)
        print(f"RMSE Test-Accuracy: {test_rmse}")

        if multi_input:
            train_prediction = model.predict((X_train, np.array(ol_train).reshape(-1, 1)))
        elif multi_input is None:
            X_model_input_train = []
            for i in range(num_feats):
                X_model_input_train.append(X_train[:, :, i])
            train_prediction = model.predict(X_model_input_train)
        else:
            train_prediction = model.predict(X_train)
        train_prediction = train_prediction.reshape(-1)
        train_preds = train_prediction / max_heats * ol_train if normalize_campaign_size else train_prediction
        train_rmse = root_mean_squared_error(y_train, train_preds.reshape(-1))
        print(f"RMSE Training-Accuracy: {train_rmse}")

        if multi_input:
            whole_prediction = model.predict((d, np.array(original_lengths).reshape(-1, 1)))
        elif multi_input is None:
            X_model_input = []
            for i in range(num_feats):
                X_model_input.append(d[:, :, i])
            whole_prediction = model.predict(X_model_input)
        else:
            whole_prediction = model.predict(d)

        whole_preds = whole_prediction.reshape(
            -1) / max_heats * original_lengths if normalize_campaign_size else whole_prediction.reshape(-1)
        whole_rmse = root_mean_squared_error(t, whole_preds)
        print(f"RMSE Whole Dataset: {whole_rmse}")

    return history, model, whole_preds, test_preds, test_rmse, train_rmse, whole_rmse


if __name__ == '__main__':

    # just get model names
    _, model_names, _ = get_model(None, 160, 60)
    models_nums = list(model_names.keys())

    # read what to do from the CL
    model_number = input("What model to you want to use:\n"
                         + "\n".join([f'Enter "{x}" to use the {model_names[x]}' for x in models_nums]) +
                         "\nEnter 'c' to compare all models\n"
                         ).strip().strip("'").strip('"')
    model_number = int(model_number) if model_number.isnumeric() else model_number

    if model_number in ['c', 'C']:
        steps = input("If you want to compare the results after a certain amount of epochs, press Enter\n"
                      "If you want to periodically compare all models while they are trained, enter the step size:\n")
        steps = int(steps) if steps.isnumeric() else None
    if model_number not in models_nums and model_number not in ['c', 'C']:
        model_number = 1

    load = input("Load existing model weights? (enter 'y' or 'n'):\n").strip().strip("'").strip('"').lower()

    if load in ["y", "yes"]:
        train = input("Do you want to continue training this model (enter 't') or evaluate its current state? "
                      "(enter anything else):\n").strip().strip("'").strip('"')
    else:
        train = "t"
    if train in [None, "t", "train"]:
        epochs = input("Enter the number of epochs to train (default = 250):\n").strip().strip("'").strip('"')
        epochs = 250 if not epochs.isnumeric() else int(epochs)

    file_paths = ['data/dataset_jsis_ladle_wall_metalzone_cut_prep_20220329_105019.pkl',
                      'data/dataset_jsis_ladle_wall_slagzone_prep_20220329_104634.pkl',
                      'data/HeartInData_07_2022.pkl', 'data/HeartOutData_07_2022.pkl', 'data/HeartData_07_2022.pkl']
    file_titles = ['Metalzone', 'Slagzone', 'Bottom Inlet', 'Bottom Outlet', 'Heart']

    if use_all_datasets:
        if selected_dataset == "voest" or combine_datasets:
            file_paths = ['data/HeartInData_07_2022.pkl', 'data/HeartOutData_07_2022.pkl', 'data/HeartData_07_2022.pkl']
            file_titles = ['Bottom Inlet', 'Bottom Outlet', 'Heart']
        elif selected_dataset == "mzsl":
            file_paths = ['data/dataset_jsis_ladle_wall_metalzone_cut_prep_20220329_105019.pkl',
                          'data/dataset_jsis_ladle_wall_slagzone_prep_20220329_104634.pkl']
            file_titles = ['Metalzone', 'Slagzone']

    iterations = len(file_paths) if use_all_datasets and not combine_datasets else 1
    file_paths = file_paths if use_all_datasets else [file_paths[file_titles.index(selected_single_dataset)]]
    file_titles = file_titles if use_all_datasets else [file_titles[file_titles.index(selected_single_dataset)]]

    # when comparing models, write the results to an excel file
    if model_number in ["c", "C"] and train in ["t", "train"] and use_all_datasets:
        with open('results.csv', 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([])
            writer_object.writerow([f'{f"{num_fold_cv_init}-fold" if cross_validation else "No"} Cross-Validation, '
                                    f'{"With" if data_aug else "No"} Data-Augmentation, '
                                    f'{"With" if normalize_campaign_size else "No"} Normalization, '
                                    f'{"With" if gauss_filter else "No"} Gauss Filter'
                                    f'{", Combined Model" if combine_datasets else ""}'
                                    f'{f" - top {use_columns} Shap features" if isinstance(use_columns, int) else f" - {use_columns} features"}'])
            writer_object.writerow(
                (['Dataset'] if combine_datasets else ['Dataset', 'Mean Label', 'Linear Regression']) +
                [model_names[x] for x in models_nums])
            f_object.close()

    # do the same for all data sets, if chosen
    for i in range(iterations):
        d, t, max_heats, num_feats, original_lengths, cols = prepare_data(file_paths[i],
                                                                          model_number=model_number if model_number not in [
                                                                              "c", "C"] else 1)
        with open('results.csv', 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(cols)
            f_object.close()

        if plot_model_architecture:
            m, _, _ = get_model(model_number if model_number not in ['c', 'C'] else 1, max_heats, num_feats)
            plot_model(m, to_file='model.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)

        if num_fold_cv_init == 'leave_one_out':
            num_fold_cv = len(d)
        else:
            num_fold_cv = num_fold_cv_init

        # compare models
        if model_number == "c" or model_number == "C":
            if steps is None:
                if plot_shap_differences:
                    for j in range(2, 25):
                        with open('results.csv', 'a', newline='') as f_object:
                            writer_object = writer(f_object)
                            writer_object.writerow([])
                            writer_object.writerow(
                                [f'{f"{num_fold_cv_init}-fold" if cross_validation else "No"} Cross-Validation, '
                                 f'{"With" if data_aug else "No"} Data-Augmentation, '
                                 f'{"With" if normalize_campaign_size else "No"} Normalization, '
                                 f'{"With" if gauss_filter else "No"} Gauss Filter'
                                 f'{", Combined Model" if combine_datasets else ""}'
                                 f'{f" - top {j} Shap features" if isinstance(j, int) else f" - {j} features"}'])
                            writer_object.writerow(
                                (['Dataset'] if combine_datasets else ['Dataset', 'Mean Label', 'Linear Regression']) +
                                [model_names[x] for x in models_nums])
                            f_object.close()

                        d, t, max_heats, num_feats, original_lengths, cols = prepare_data(file_paths[i],
                                                                                          change_use_columns=j)  # TODO remove test
                        try:
                            train_and_compare_models(d, t, file=file_titles[i])
                        except Exception as e:
                            with open('results.csv', 'a', newline='') as f_object:
                                writer_object = writer(f_object)
                                writer_object.writerow(["Not possible with this model", e])
                                f_object.close()
                else:
                    train_and_compare_models(d, t, file=file_titles[i])
            else:
                show_and_compare_training_progression(d, t)
        else:
            if train in ["t", "train"]:
                train_single_model(model_number, cross_validation, d, t)
            else:
                model, model_names, multi_input = get_model(model_number, max_heats, num_feats)
                model_name = model_names[model_number]
                if exists(f'weights/{model_name}_model_weights.h5') and load in ["y", "yes"]:
                    model.load_weights(f'weights/{model_name}_model_weights.h5')
                    predictions = model.predict(d)
                    rmse = calculate_RMSE_whole_set(predictions)
                    show_barchart(predictions, model_names[model_number], f"Whole-Set-RMSE: {rmse:.2f}", t)
                    show_residuals(predictions, model_names[model_number], f"Whole-Set-RMSE: {rmse:.2f}", t)
                else:
                    print("model weights do not exist!")
