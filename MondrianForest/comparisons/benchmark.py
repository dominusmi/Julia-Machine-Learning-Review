#!/usr/bin/env python

import numpy as np
import pprint as pp     # pretty printing module
from matplotlib import pyplot as plt        # required only for plotting results
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal
from mondrianforest import process_command_line, MondrianForest

import time

PLOT = False

settings = process_command_line()
# Resetting random seed
reset_random_seed(settings)

# Loading data
data = load_data(settings)

# print data
# import pandas as pd
#
# test = pd.DataFrame(data['x_test'],data['y_test'])
# train = pd.DataFrame(data['x_train'], data['y_train'])
#
# print test
#
# test.to_csv('dna_test.csv', header=False)
# train.to_csv('dna_train.csv', header=False)

param, cache = precompute_minimal(data, settings)

mf = MondrianForest(settings, data)


for idx_minibatch in range(settings.n_minibatches):
    train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
    if idx_minibatch == 0:
        # Batch training for first minibatch
        start = time.time()
        mf.fit(data, train_ids_current_minibatch, settings, param, cache)
        end = time.time()
    else:
        # Online update
        start = time.time()
        mf.partial_fit(data, train_ids_current_minibatch, settings, param, cache)
        end = time.time()
    # Evaluate
    weights_prediction = np.ones(settings.n_mondrians) * 1.0 / settings.n_mondrians
    train_ids_cumulative = data['train_ids_partition']['cumulative'][idx_minibatch]
    pred_forest_train, metrics_train = \
        mf.evaluate_predictions(data, data['x_train'][train_ids_cumulative, :], \
        data['y_train'][train_ids_cumulative], \
        settings, param, weights_prediction, False)
    pred_forest_test, metrics_test = \
        mf.evaluate_predictions(data, data['x_test'], data['y_test'], \
        settings, param, weights_prediction, False)
    name_metric = settings.name_metric     # acc or mse
    metric_train = metrics_train[name_metric]
    metric_test = metrics_test[name_metric]
    print metric_test, end-start
