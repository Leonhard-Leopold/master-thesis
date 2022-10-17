from tensorflow.keras import layers, models, metrics
from tensorflow.keras import regularizers, initializers
from keras import backend as K
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Flatten, MaxPooling1D, Input, MaxPooling2D


loss_function = None
focal_loss_gamma = 1


def init_loss_function(loss, gamma):
    global loss_function, focal_loss_gamma
    loss_function = loss
    focal_loss_gamma = gamma


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def focal_loss(y_true, y_pred):
    l2 = K.square(K.abs(y_pred-y_true))
    return K.pow(l2, focal_loss_gamma+2)


# Here different models were manually tested. This was done to find the best model
# Different models work best for different dataset


def final_model_ms_sl_dataset(model_number, max_heats, num_feats, outputs=1):
    model_names = {
        1: '2D-CNN 2 layers',
    }
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names, False


def final_models_voest_dataset(model_number, max_heats, num_feats, outputs=1):

    model_names = {
        1: 'GRU ResNet',
    }

    model = models.Sequential()
    if model_number == 1:
        first_input = layers.Input(shape=(max_heats, num_feats))
        first_dense = layers.GRU(32, activation='relu', return_sequences=True)(first_input)
        first_dense = layers.GRU(32, activation='relu', return_sequences=True)(first_dense)
        second_dense = layers.GRU(32, activation='relu', return_sequences=True)(first_dense)
        second_dense = layers.GRU(32, activation='relu', return_sequences=True)(second_dense)
        merged_third = layers.concatenate([first_dense, second_dense])
        merged_third = layers.GRU(32, activation='relu')(merged_third)
        merged_third = layers.Dense(32, activation='relu')(merged_third)
        third_dense = layers.Dense(outputs)(merged_third)
        model = models.Model(inputs=[first_input], outputs=third_dense)
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names, False


def final_models_voest_dataset_2_inputs(model_number, max_heats, num_feats, outputs=1):
    model_names = {
        1: 'GRU resnet - 2 inputs'
    }
    model = models.Sequential()
    if model_number == 3:
        first_input = layers.Input(shape=(max_heats, num_feats))
        first_dense = layers.GRU(64, return_sequences=True)(first_input)
        first_dense = layers.GRU(64)(first_dense)

        second_input = layers.Input(shape=(1, 1))
        single_dense = layers.Flatten()(second_input)
        single_dense_32 = layers.Dense(64)(single_dense)

        second_dense = layers.GRU(128, return_sequences=True)(first_input)
        second_dense = layers.GRU(128, return_sequences=True)(second_dense)
        second_dense = layers.GRU(64)(second_dense)

        merged_third = layers.concatenate([first_dense, second_dense, single_dense_32])
        first_dense = layers.Reshape((64, 1))(first_dense)
        second_dense = layers.Reshape((64, 1))(second_dense)
        first_dense = layers.GRU(32)(first_dense)
        second_dense = layers.GRU(32)(second_dense)
        merged_third = layers.Dense(32)(merged_third)
        merged_fourth = layers.concatenate([first_dense, second_dense, merged_third])
        third_dense = layers.Dense(outputs)(merged_fourth)

        model = models.Model(inputs=[first_input, second_input], outputs=third_dense)
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
        return model, model_names, True
    return model, model_names, False


# default model to determine SHAP values
def shap_model(model_number, max_heats, num_feats, outputs=1):
    model_names = {
        1: '6l - 6x6',
    }
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names, False


def initial_models(model_number, max_heats, num_feats, outputs=1):
    model_names = {1: '2D CNN',
                   2: 'Deep 2D CNN',
                   3: '1D CNN',
                   4: 'Deep 1D CNN',
                   5: '2D Conv ResNet',
                   6: 'Dense',
                   7: 'Deep Dense',
                   8: 'LSTM',
                   9: 'GRU'}

    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 3:
        model.add(layers.Conv1D(num_feats, num_feats, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_feats, activation='relu'))
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 4:
        model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 5:
        input = layers.Input(shape=(max_heats, num_feats, 1))
        layer1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)
        layer2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)
        layer3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)
        layer_out = layers.concatenate([layer1, layer2], axis=-1)
        layer4 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(layer_out)
        layer_out = layers.concatenate([layer3, layer4], axis=-1)
        layer_out = layers.Conv2D(64, (3, 3), activation='relu')(layer_out)
        layer_out = layers.MaxPool2D(2, 2)(layer_out)
        layer_out = layers.Flatten()(layer_out)
        layer_out = layers.Dense(64, activation='relu')(layer_out)
        layer_out = layers.Dense(32, activation='relu')(layer_out)
        output = layers.Dense(outputs)(layer_out)
        model = models.Model(inputs=input, outputs=output)
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 6:
        model.add(layers.Dense(64, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(outputs))
        model.add(layers.Flatten())
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 7:
        model.add(layers.Dense(64, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(outputs))
        model.add(layers.Flatten())
        model.add(layers.Dense(outputs))
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 8:
        input = layers.Input(shape=(max_heats, num_feats))
        layer = layers.LSTM(64, return_sequences=True)(input)
        layer = layers.LSTM(64, return_sequences=True)(layer)
        layer = layers.LSTM(64)(layer)
        layer = layers.Dense(32, activation='relu')(layer)
        layer = layers.Dense(outputs)(layer)
        model = models.Model(inputs=[input], outputs=layer)
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 9:
        input = layers.Input(shape=(max_heats, num_feats))
        layer = layers.GRU(64, return_sequences=True)(input)
        layer = layers.GRU(64, return_sequences=True)(layer)
        layer = layers.GRU(64)(layer)
        layer = layers.Dense(32, activation='relu')(layer)
        layer = layers.Dense(outputs)(layer)
        model = models.Model(inputs=[input], outputs=layer)
        model.compile(optimizer='adam', loss=loss_function, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names, False
