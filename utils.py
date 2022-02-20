import math
import os
import pickle

import numpy as np
from prettytable import PrettyTable


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def get_normalized_features(X):
    means = np.mean(X, axis=(0, 1))  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, 1, -1))
    stds = np.std(X, axis=(0, 1))  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, 1, -1))
    return X, means, stds


def masked_MAE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.mean(np.absolute(pred - target))


def masked_MSE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.mean((pred - target) ** 2)


def masked_RMSE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.sqrt(np.mean((pred - target) ** 2))


def masked_MAPE(pred, target):
    valid_indices = np.nonzero(target)
    pred = pred[valid_indices]
    target = target[valid_indices]
    return np.mean(np.absolute((pred - target) / (target + 0.1))) * 100


def elapsed_time_format(total_time):
    hour = 60 * 60
    minute = 60
    if total_time < 60:
        return f"{math.ceil(total_time):d} secs"
    elif total_time > hour:
        hours = divmod(total_time, hour)
        return f"{int(hours[0]):d} hours, {elapsed_time_format(hours[1])}"
    else:
        minutes = divmod(total_time, minute)
        return f"{int(minutes[0]):d} mins, {elapsed_time_format(minutes[1])}"


def model_summary(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    return table, total_params


def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result
