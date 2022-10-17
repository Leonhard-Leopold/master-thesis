import numpy as np
import random
import pickle
from scipy import ndimage
from shap_values import get_shap_values
from main import *
import pandas as pd
from sklearn.utils import shuffle


# load data, apply gauss filter, normalize it, pad it to the same length
def prepare_data(filename, all=False, change_use_columns=None, model_number=1):
    global gauss_filter, normalize_data, zero_padding, classification, use_columns, data_aug_shuffle

    if change_use_columns is not None:
        use_columns = change_use_columns

    if combine_datasets and not all:
        data, HS_targets_prepared, column_names = combine_hard_datasets()
    else:
        data, HS_targets_prepared, column_names = load_data(filename, all=all, model_number=model_number)

    if gauss_filter and not data_aug_shuffle:
        data = apply_gauss_filter(data)

    # get dimensions
    max_heats = max(d.shape[0] for d in data)
    num_feats = data[0].shape[1]

    if normalize_data:
        for i, d in enumerate(data):
            data[i] = np.nan_to_num(data[i] / np.max(data[i], axis=0))

    original_lengths = []
    for i, d in enumerate(data):
        original_lengths.append(d.shape[0])
        if max_heats != d.shape[0]:
            pad = np.zeros(len(d[-1])) if zero_padding else d[-1]
            data[i] = np.append(d, np.array([pad] * (max_heats - d.shape[0])), axis=0)

    d, t = np.array(data).reshape((-1, max_heats, num_feats, 1)), np.array(HS_targets_prepared)

    if classification:
        t = classify_targets(t, original_lengths)

    # during the load_data function the shap values are not calculated but the datasets are combined.
    if combine_datasets and not all and isinstance(use_columns, int):
        dataset = d, t, max_heats, num_feats, original_lengths, column_names
        shap_values, shap_values_reshaped, X_test, column_names, max_indexes = get_shap_values(
            shap_model, 1, dataset, False, random_state)

        top_columns = np.array(column_names)[max_indexes][:use_columns]
        unneeded_columns = [e for e in column_names if e not in top_columns]
        reduced_data = []
        for campaign in d:
            c = pd.DataFrame(data=campaign.reshape(campaign.shape[:-1]), columns=column_names).drop(unneeded_columns, axis=1, errors="ignore")
            reduced_data.append(np.array(c.values.tolist()))
        d = reduced_data
        column_names = top_columns
        num_feats = use_columns

    return d, t, max_heats, num_feats, original_lengths, column_names


# currently unused!
# Creating one-hot-encoded labels for high, middle and low targets
def classify_targets(targets, original_lengths):
    stds = 0.65
    ta = targets / original_lengths if True else targets
    new_targets = []
    for ta_ in ta:
        if ta_ < np.mean(ta) - np.std(ta) * stds:
            new_targets.append([1, 0, 0])
        elif ta_ > np.mean(ta) + np.std(ta) * stds:
            new_targets.append([0, 0, 1])
        else:
            new_targets.append([0, 1, 0])
    return new_targets


# function used while combining datasets - Finds the index of a mostly similar object
def get_index(object, arr):
    index = None
    for i, a in enumerate(arr):
        difference = np.array(object == a).reshape(-1).tolist().count(False)
        if len(a) != len(object):
            continue
        if difference < 50:
            return i
    return index


# combining datasets to only train one model for similar datasets
def combine_hard_datasets():
    data1, targets1, _ = load_data('data/BottomInletData_04_2022.pkl', all=True)
    data2, targets2, _ = load_data('data/BottoOutletData_04_2022.pkl', all=True)
    data3, targets3, column_names = load_data('data/HeartData_04_22.pkl', all=True)

    combined_data, combined_targets = data1, np.array(targets1).reshape(-1, 1).tolist()

    for i, d in enumerate(data2):
        index = get_index(d, combined_data)
        if index == None:
            combined_data.append(d)
            combined_targets.append([targets2[i]])
        else:
            combined_targets[index].append(targets2[i])

    for i, d in enumerate(data3):
        index = get_index(d, combined_data)
        if index == None:
            combined_data.append(d)
            combined_targets.append([targets3[i]])
        else:
            combined_targets[index].append(targets3[i])

    data, targets = [], []
    for i, t in enumerate(combined_targets):
        if len(t) == 3:
            data.append(combined_data[i])
            targets.append(t)

    return data, targets, column_names


# applying a 1D Gauss Filter to all (manually selected) features
def apply_gauss_filter(data):
    sigma = 10
    filtered = []

    # manually determined
    columns_to_filter = ['Leerzeit [min]', 'Abstgew. inkl. Leg./Zuschl.[t]', 'RS-Index',
                         'Abstichdauer [min]', 'Abst.Zugabe Kohle [kg]',
                         'Entschwefelung Turbokalk [kg]', 'Entschwefelung Tonerde [kg]',
                         'Entschwefelung Kalk [kg]', '1.Pfa-Temp nach Abst. [C°]',
                         'Heizenergie [MWh]', 'Start RH-Temp [C°]', 'Ende RH-Temp [C°]',
                         'TR max. SP-Menge [m3/h]', 'BR O2 Mng. vor BBeg [m3]', 'FL-Kühlung TR-EL Temp [C°]',
                         'FL-Kühlung TR-AL Temp [C°]', 'FL-Kühlung UG Temp [C°]', 'Al Menge [kg/t RStfl.]',
                         '[C] vor BBeg [%]', '[C] nach BBeg [%]', '[C] Fertigprobe [%]',
                         '[Si] vor BBeg [%]', '[Si] nach BBeg [%]', '[Si] Fertigprobe [%]',
                         '[Mn] vor BBeg [%]', '[Mn] nach BBeg [%]', '[Mn] Fertigprobe [%]',
                         '[P] vor BBeg [%]', '[P] nach BBeg [%]', '[P] Fertigprobe [%]',
                         '[Al] vor BBeg [%]', '[Al] nach BBeg [%]', '[Al] Fertigprobe [%]',
                         '[Ca] vor BBeg [%]', '[Ca] nach BBeg [%]', '[Ca] Fertigprobe [%]',
                         '[N] vor BBeg [%]', '[N] nach BBeg [%]', '[N] Fertigprobe [%]',
                         'Behandlungsdauer [min]', 'SGR BBeg Druck AVG [bar]', 'SGR 20.min Druck AVG [bar]',
                         'SGR 20.min RV-Stellung AVG [%]',
                         'Tap Weight [t]', 'EAFTap-LFStart [min]', 'LF Process Time [min]',
                         'LF P ON Time [min]', 'LF Argon Cons. [Nm3]', 'LF Argon Cons. [Nm3/MT]',
                         'LF Nitrogen Cons. [Nm3]', 'LF Nitrogen Cons. [Nm3/MT]', 'LF Start temp. [C]',
                         'LF Power [kWh]', 'LF Power [kWh/MT]', 'LF Ca wire [kg]',
                         'LF Lime [kg]', 'Tap+LF Lime [kg]', 'Tap+LF Dolo [kg]',
                         'Tap+LF SPL [kg]', 'Tap+LF Al [kg]', 'LF End C [%]',
                         'LF End Si [%]', 'LF End Mn [%]', 'LF End Al [%]',
                         'LF End N [ppm]', 'LF End CaO [%]', 'LF End MgO [%]',
                         'LF End FeO [%]', 'LF End MnO [%]', 'LF End Al2O3 [%]',
                         'LF End SiO2 [%]', 'LF End S [%].1', 'LF B2',
                         'LF B3', 'EAF Temp. [C]']

    from matplotlib import pyplot as plt
    for d in data:
        help = []
        for i in range(len(d[0])):
            if cols[i] in columns_to_filter:
                help.append(ndimage.gaussian_filter1d(d[:, i], sigma))
            else:
                help.append(d[:, i])
        help = np.array(help).transpose()
        filtered.append(help)
    return filtered


# load data from file and remove unneeded columns
def load_data(file, all=False, model_number=1):
    global cols, use_columns, get_best_model, models_nums, get_model

    with open(file, 'rb+') as f:
        loaded_data = pickle.load(f)
        if len(loaded_data) == 4:
            [list_prod_prepared, HS_targets_prepared, list_abstellgrund_prepared, list_initial] = loaded_data
        elif len(loaded_data) == 2:
            [list_prod_prepared, HS_targets_prepared] = loaded_data
        f.close()

    # remove empty and unneeded columns
    data = []
    excluded = ['Heat ID', 'Campaign', 'TR-Pflege Zeitpunkt [Dtm]', 'TR-Pflege Menge [kg]', 'SGR RV-Stellung AVG [%]',
                'SGR BBeg Menge AVG [m3/h]',

                'Heat', 'Ladle MZ heats', 'LF CaSi wire [kg]', 'LF Dolo [kg]', 'LF Spar [kg]', 'LF SPL [kg]',
                'Tap+LF Spar [kg]', 'LF End Ca [ppm]']
    unneeded_columns = find_unneeded_features(list_prod_prepared, excluded)
    unneeded_columns.extend(excluded)

    all_cols = list_prod_prepared[0].columns.tolist()
    if all:
        pass
    elif use_columns == 'best':
        # considered to be the most important features
        important_columns = ['Leerzeit [min]', 'unberuhigter Abstich', 'Entschwefelung Turbokalk [kg]',
                             'Entschwefelung CaC2 [kg]', 'Heizenergie [MWh]', 'O2 Menge in Behand. [m3]',
                             'O2 Menge vor Behand. [m3]', 'BR O2 Mng. vor BBeg [m3]', 'BR Erdgas Mng. vor BBeg [m3]',
                             'Si Menge [kg/t RStfl.]', '[Si] Fertigprobe [%]', '[Mn] Fertigprobe [%]',
                             '[Al] Fertigprobe [%]', '[Ca] vor BBeg [%]', '[Ca] nach BBeg [%]', '[Ca] Fertigprobe [%]',
                             'Behandlungsdauer [min]']
        unneeded_columns = [e for e in all_cols if e not in important_columns]
    elif use_columns == 'best + extended':
        # considered to be the most important features + other possibly important features
        important_columns = ['Leerzeit [min]', 'unberuhigter Abstich', 'Entschwefelung Turbokalk [kg]',
                             'Entschwefelung CaC2 [kg]', 'Heizenergie [MWh]', 'O2 Menge in Behand. [m3]',
                             'O2 Menge vor Behand. [m3]', 'BR O2 Mng. vor BBeg [m3]', 'BR Erdgas Mng. vor BBeg [m3]',
                             'Si Menge [kg/t RStfl.]', '[Si] Fertigprobe [%]', '[Mn] Fertigprobe [%]',
                             '[Al] Fertigprobe [%]', '[Ca] vor BBeg [%]', '[Ca] nach BBeg [%]', '[Ca] Fertigprobe [%]',
                             'Behandlungsdauer [min]', 'Abst.Zugabe Kalk [kg]', 'Abst.Zugabe Schlackeng. [kg]',
                             'Abst.Zugabe Rohmagnesit [kg]', 'Entschwefelung Tonerde [kg]', 'Entschwefelung Kalk [kg]',
                             'Start RH-Temp [C°]', 'Ende RH-Temp [C°]', 'O vor RH [ppm]', 'O vor Al an RH [ppm]',
                             'Al Menge [kg/t RStfl.]', '[P] Fertigprobe [%]']
        unneeded_columns = [e for e in all_cols if e not in important_columns]
    elif isinstance(use_columns, int):
        # use shap values to find the most impactful features
        try:
            # Use currently used model to determine SHAP values - not possible for all types of models.
            # Thus, fallback is used

            # shap_values, shap_values_reshaped, X_test, column_names, max_indexes = get_shap_values(
            #      get_model, model_number, file, False, random_state)

            # designated default model to determine SHAP values
            shap_values, shap_values_reshaped, X_test, column_names, max_indexes = get_shap_values(
               shap_model, 1, file, False, random_state)
        except Exception as e:
            print("This model cannot be used to determine Shap values. A fallback model was used. \n Error", e)
            shap_values, shap_values_reshaped, X_test, column_names, max_indexes = get_shap_values(
                shap_model, 1, file, False, random_state)

        top_columns = np.array(column_names)[max_indexes][:use_columns]
        unneeded_columns = [e for e in all_cols if e not in top_columns]

    for campaign in list_prod_prepared:
        c = campaign.drop(unneeded_columns, axis=1, errors="ignore") # if not all else campaign
        cols = c.columns.tolist()
        data.append(np.array(c.values.tolist()))
    return data, HS_targets_prepared, cols


# finding features that do not deviate over time to be removed later
def find_unneeded_features(d, excluded):
    unneeded = []
    dict = {}
    for cat in d[0].columns:
        dict[cat] = []
    for campaign in d:
        for cat in d[0].columns:
            if cat in excluded:
                continue
            dict[cat].append(campaign[cat].to_numpy().std())
    for cat in d[0].columns:
        if np.array(dict[cat]).std() == 0:
            unneeded.append(cat)
    return unneeded


# mix-up data augmentation using a beta distribution
def data_augmentation(data, targets, lengths=None):
    global aug_amount
    new_data, new_labels, new_lengths = [], [], []
    for i in range(len(data) * aug_amount):
        lam = np.random.beta(0.2, 0.2)  # beta distribution
        a, b = random.randint(0, len(data) - 1), random.randint(0, len(data) - 1)
        new_data.append(lam * data[a] + (1 - lam) * data[b])
        new_labels.append(lam * targets[a] + (1 - lam) * targets[b])
        new_lengths.append(max(lengths[a], lengths[b]))

    xx = np.concatenate([data, new_data])
    yy = np.concatenate([targets, new_labels])
    ll = np.concatenate([lengths, new_lengths])
    return xx, yy, ll


# mix-up data augmentation using a beta distribution
def shuffle_data_augmentation(data, targets, lengths=None):
    global aug_amount
    new_data, new_labels, new_lengths = [], [], []
    for i, d in enumerate(data):
        for j in range(aug_amount):
            d_ = data[i]
            np.random.shuffle(d_)
            new_data.append(d_)
            new_labels.append(targets[i] + np.random.uniform(-10.0, 10.0))
            new_lengths.append(lengths[i])

    xx = np.concatenate([data, new_data])
    yy = np.concatenate([targets, new_labels])
    ll = np.concatenate([lengths, new_lengths])

    xx, yy, ll = shuffle(np.array(xx), np.array(yy), np.array(ll))

    return xx, yy, ll