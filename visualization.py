import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# the progression of the RMSE over epochs is plotted
def show_RMSE_over_epochs(history, c, model_name, show=False, type="both"):
    global colors
    c = c % 7
    color = colors[c]
    if type == "both":
        plt.plot(history.history['root_mean_squared_error'], label=f'{model_name} - RMSE', color='r')
        plt.plot(history.history['val_root_mean_squared_error'], label=f'{model_name} - Validation-RMSE', color='g')
        plt.title(f'RMSE over Epochs')
    elif type == "train":
        plt.plot(history.history['root_mean_squared_error'], label=f'{model_name}', color=color)
        plt.title(f'RMSE of the Training Set over Epochs')

    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    if show:
        plt.show()


# plotting barchart that compares predictions to the targets
def show_barchart(predictions, model_name, rmse, tar):
    n_groups = len(predictions)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    # checking if a combined models for multiple datasets were used
    if isinstance(predictions[0].tolist(), float):
        plt.bar(index, predictions.reshape(-1), bar_width, alpha=opacity, color='b', label='Predictions')
        plt.bar(index + bar_width, tar, bar_width, alpha=opacity, color='g', label='Targets')
        plt.xlabel('Campaigns')
        plt.ylabel('Wear [mm]')
        plt.title(f'Wear Predictions - {model_name}\n{rmse}')
        plt.legend()
        plt.show()
    else:
        for i in range(3):
            plt.bar(index, predictions[:, i], bar_width, alpha=opacity, color='b', label='Predictions')
            plt.bar(index + bar_width, tar[:, i], bar_width, alpha=opacity, color='g', label='Targets')
            plt.xlabel('Campaigns')
            plt.ylabel('Wear [mm]')
            plt.title(f'Wear Predictions - {model_name}\n{rmse}')
            plt.legend()
            plt.show()


# plotting the residuals (difference between each prediction and its target)
def show_residuals(predictions, model_name, rmse, tar):
    n_groups = len(predictions)

    # checking if a combined models for multiple datasets were used
    if isinstance(predictions[0].tolist(), float):
        index = np.arange(n_groups)
        plt.bar(index, predictions.reshape(-1) - tar, label='Residuals')
        plt.xlabel('Campaigns')
        plt.ylabel('Residuals [mm]')
        plt.title(f'Residuals between Predictions and Targets - {model_name}\n{rmse}')
        plt.legend()
        plt.show()
    else:
        for i in range(3):
            index = np.arange(n_groups)
            plt.bar(index, predictions[:, i] - tar[:, i], label='Residuals')
            plt.xlabel('Campaigns')
            plt.ylabel('Residuals [mm]')
            plt.title(f'Residuals between Predictions and Targets - {model_name}\n{rmse}')
            plt.legend()
            plt.show()


# plotting the input data using T-SNE or PCA
def plot_input_data(data, targets, model_name, method="tsne", average=True):
    if average:
        x = np.array([[np.average(np.array(data[y])[:, x]) for x in range(len(data[0][0]))] for y in range(len(data))])
    else:
        x = np.array([np.hstack(x) for x in data[:, :, :, 0]])

    if method == "tsne":
        x_reduced = TSNE(n_components=2, init='random').fit_transform(x)
    else:
        x_reduced = PCA(n_components=2).fit_transform(x)

    points = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=targets)
    plt.colorbar(points)
    plt.title(f"{model_name} \nInput data - reduced using {method.upper()} {' - features averaged' if average else ''}")
    plt.show()
    print("")
