from tensorflow.keras import layers, models, metrics
from tensorflow.keras import regularizers, initializers
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def select_initial_models(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN', 2: 'Deep 2D CNN', 3: '1D and 2D CNN', 4: '1D CNN', 5: '2D Conv ResNet', 6: 'Dense',
                   7: 'Deep Dense'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 3:
        model.add(layers.Conv1D(64, 3, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv1D(32, 3, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 4:
        model.add(layers.Conv1D(num_feats, num_feats, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_feats, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 5:
        input = layers.Input(shape=(max_heats, num_feats, 1))
        layer1 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(input)
        layer3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input)
        layer5 = layers.Conv2D(64, (5, 5), padding='same', activation='relu')(input)
        layer_out = layers.concatenate([layer1, layer3, layer5], axis=-1)
        layer_out = layers.Flatten()(layer_out)
        layer_out = layers.Dense(64, activation='relu')(layer_out)
        layer_out = layers.Dense(32, activation='relu')(layer_out)
        output = layers.Dense(1)(layer_out)
        model = models.Model(inputs=input, outputs=output)
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 6:
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 7:
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_models_with_kernel_regularisation(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN', 2: '2D CNN - kernel l1_l2 reg', 3: '2D CNN - kernel l1 reg',
                   4: '2D CNN - kernel l2 reg', 5: '2D CNN - kernel l2 partly reg'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2()))
        model.add(layers.Dense(1, kernel_regularizer=regularizers.l1_l2()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1()))
        model.add(layers.Dense(1, kernel_regularizer=regularizers.l1()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.Dense(1, kernel_regularizer=regularizers.l2()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    return model, model_names


def select_models_with_bias_regularisation(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN', 2: '2D CNN - bias l1_l2 reg', 3: '2D CNN - bias l1 reg', 4: '2D CNN - bias l2 reg',
                   5: '2D CNN - bias l2 partly reg'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_regularizer=regularizers.l1_l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1_l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1_l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1_l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer=regularizers.l1_l2()))
        model.add(layers.Dense(32, activation='relu', bias_regularizer=regularizers.l1_l2()))
        model.add(layers.Dense(1, bias_regularizer=regularizers.l1_l2()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_regularizer=regularizers.l1(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.Dense(32, activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.Dense(1, bias_regularizer=regularizers.l1()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1, bias_regularizer=regularizers.l2()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_models_with_activity_regularisation(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN', 2: '2D CNN - activity l1_l2 reg', 3: '2D CNN - activity l1 reg',
                   4: '2D CNN - activity l2 reg', 5: '2D CNN - activity l2 partly reg'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', activity_regularizer=regularizers.l1_l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l1_l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l1_l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l1_l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.l1_l2()))
        model.add(layers.Dense(32, activation='relu', activity_regularizer=regularizers.l1_l2()))
        model.add(layers.Dense(1, activity_regularizer=regularizers.l1_l2()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', activity_regularizer=regularizers.l1(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l1()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.l1()))
        model.add(layers.Dense(32, activation='relu', activity_regularizer=regularizers.l1()))
        model.add(layers.Dense(1, activity_regularizer=regularizers.l1()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', activity_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', activity_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', activity_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', activity_regularizer=regularizers.l2()))
        model.add(layers.Dense(1, activity_regularizer=regularizers.l2()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_models_with_initializers(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN', 2: '2D CNN - ND Kernel Init', 3: '2D CNN - ND Bias Init',
                   4: '2D CNN - ND Kernel and Bias Init'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(
            layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01),
                          input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(
            layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(
            layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(
            layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(32, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(32, activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                                kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                                kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                                kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                                kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                               kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(32, activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                               kernel_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    return model, model_names


def select_models_with_dropout(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN', 2: '2D CNN - 0.2 Dropout', 3: '2D CNN - 0.2 Dropout (bottom)',
                   4: '2D CNN - 0.2 Dropout (top)', 5: '2D CNN - 0.1 Dropout (bottom)', 6: '2D CNN - 0.1 Dropout (top)',
                   7: '2D CNN - 0.1 Dropout'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 6:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 7:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_models_with_different_nums_of_layers(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN - 1 layer', 2: '2D CNN - 2 layer', 3: '2D CNN - 3 layer', 4: '2D CNN - 4 layer',
                   5: '2D CNN - 5 layer', 6: '2D CNN - 6 layer', 7: '2D CNN - 7 layer', 8: '2D CNN - 8 layer'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 6:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 7:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 8:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(18, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_models_with_max_pooling(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN - no max pooling', 2: '2D CNN - 1 max pooling layer', 3: '2D CNN - 3 max pooling layers'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_models_with_different_nums_of_units(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN - 16 units per layer', 2: '2D CNN - 32 units per layer',
                   3: '2D CNN - 64 units per layer', 4: '2D CNN - 128 units per layer',
                   5: '2D CNN - 256 units per layer'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_models_with_different_kernel_sizes(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN - kernel size 2x2', 2: '2D CNN - kernel size 3x3', 3: '2D CNN - kernel size 4x4',
                   4: '2D CNN - kernel size 5x5', 5: '2D CNN - kernel size 6x6', 6: '2D CNN - kernel size 7x7'}
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(32, (4, 4), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (4, 4), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(32, (6, 6), padding="same", activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 6:
        model.add(layers.Conv2D(32, (7, 7), padding="same", activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((7, 7)))
        model.add(layers.Conv2D(64, (7, 7), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (7, 7), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (7, 7), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_best_models_iteration1(model_number, max_heats, num_feats):
    model_names = {1: '2D CNN - 6 layers ',
                   2: '2D CNN - partially l2 kernel regularized',
                   3: '2D CNN - l2 bias regularized',
                   4: '2D CNN - l1 bias regularized',
                   5: '2D CNN - ND bias initializer',
                   6: '2D CNN - 0.5 Dropout (bottom layers)',
                   7: '2D CNN - 0.1 Dropout (top layers)',
                   8: '2D CNN - 2 layers',
                   9: '2D CNN - 3 layers',
                   10: '2D CNN - 4 layers',
                   11: '2D CNN - 7 layers',
                   12: '2D CNN - 1 max pooling layer',
                   13: '2D CNN - 128 units per layer',
                   14: '2D CNN - kernel size 4x4',
                   15: '2D CNN - kernel size 6x6'
                   }
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 2:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    elif model_number == 3:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1, bias_regularizer=regularizers.l2()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    elif model_number == 4:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_regularizer=regularizers.l1(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.Dense(32, activation='relu', bias_regularizer=regularizers.l1()))
        model.add(layers.Dense(1, bias_regularizer=regularizers.l1()))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    elif model_number == 5:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(32, activation='relu', bias_initializer=initializers.RandomNormal(stddev=0.01)))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    elif model_number == 6:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 7:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.1))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 8:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 9:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 10:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 11:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 12:
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 13:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 14:
        model.add(layers.Conv2D(32, (4, 4), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (4, 4), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (4, 4), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 15:
        model.add(layers.Conv2D(32, (6, 6), padding="same", activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    return model, model_names


def select_best_models_iteration2(model_number, max_heats, num_feats):
    model_names = {
        1: '6 layers',
        2: '6 layers - partially l2 kernel & l2 bias regularized',
        3: '6 layers - partially l2 kernel & l2 bias regularized & less max pooling',
        4: '4 layers',
        5: '4 layers - partially l2 kernel & l2 bias regularized',
        6: '2 layers',
        7: '2 layers - partially l2 kernel & l2 bias regularized',

        8: '6 layers - kernel size 6x6',
        9: '6 layers - kernel size 6x6 - partially l2 kernel & l2 bias regularized',
        10: '6 layers - kernel size 6x6 - partially l2 kernel & l2 bias regularized & less max pooling',
        11: '4 layers - kernel size 6x6',
        12: '4 layers - kernel size 6x6 - partially l2 kernel & l2 bias regularized',
        13: '2 layers - kernel size 6x6',
        14: '2 layers - kernel size 6x6 - partially l2 kernel & l2 bias regularized',

        15: '6 layers - 128 units',
        16: '6 layers - 128 units - partially l2 kernel & l2 bias regularized',
        17: '6 layers - 128 units - partially l2 kernel & l2 bias regularized & less max pooling',
        18: '4 layers - 128 units',
        19: '4 layers - 128 units - partially l2 kernel & l2 bias regularized',
        20: '2 layers - 128 units',
        21: '2 layers - 128 units - partially l2 kernel & l2 bias regularized',
    }

    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 2:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 3:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 6:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 7:
        model.add(layers.Conv2D(64, (3, 3), bias_regularizer=regularizers.l2(), activation='relu',
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'), kernel_regularizer=regularizers.l2(),
                  bias_regularizer=regularizers.l2())
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    if model_number == 8:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 9:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 10:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 11:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 12:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 13:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 14:
        model.add(layers.Conv2D(64, (6, 6), bias_regularizer=regularizers.l2(), activation='relu',
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'), kernel_regularizer=regularizers.l2(),
                  bias_regularizer=regularizers.l2())
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    if model_number == 15:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 16:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    elif model_number == 17:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 18:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 19:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(),
                               bias_regularizer=regularizers.l2()))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 20:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 21:
        model.add(layers.Conv2D(128, (3, 3), bias_regularizer=regularizers.l2(), activation='relu',
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'), kernel_regularizer=regularizers.l2(),
                  bias_regularizer=regularizers.l2())
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])

    return model, model_names


def select_best_models_iteration3(model_number, max_heats, num_feats):
    model_names = {
        1: '2l - 3x3',
        2: '2l - 3x3 - 128u',
        3: '2l - 6x6 - 128u',
        4: '4l - 6x6 - reg',
        5: '6l - 6x6',
        6: '6l - 128u',
    }
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
        model.add(layers.Conv2D(128, (6, 6), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 4:
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2(),
                                input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (6, 6), activation='relu', bias_regularizer=regularizers.l2()))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 5:
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
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 6:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def select_best_models_iteration4(model_number, max_heats, num_feats):
    model_names = {
        1: '2l - 3x3',
        2: '2l - 3x3 - 128u',
        3: '6l - 6x6',
    }
    model = models.Sequential()
    if model_number == 1:
        model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 2:
        model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(max_heats, num_feats, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    if model_number == 3:
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
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names


def get_best_model(model_number, max_heats, num_feats):
    model_names = {
        1: '6l - 6x6',
    }
    model = models.Sequential()
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
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[metrics.RootMeanSquaredError()])
    return model, model_names
