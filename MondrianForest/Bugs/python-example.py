#!/usr/bin/env python
import numpy as np
import pprint as pp     # pretty printing module
from matplotlib import pyplot as plt        # required only for plotting results
from mondrianforest_utils import load_data, reset_random_seed, precompute_minimal
from mondrianforest import process_command_line, MondrianForest

settings = process_command_line()
print 'Current settings:'
pp.pprint(vars(settings))

# Resetting random seed
reset_random_seed(settings)

# Loading data
data = load_data(settings)
print "Data: ", data
print type(settings)

param, cache = precompute_minimal(data, settings)

mf = MondrianForest(settings, data)

data['x_test']
print(data)

train_ids_current_minibatch = data['train_ids_partition']['current'][0]
print train_ids_current_minibatch.shape

print "First batch train on 5 data points"
mf.fit(data, train_ids_current_minibatch[0:5], settings, param, cache)
print mf.forest[0].counts
print "\nNow the extension\n"
mf.partial_fit(data, train_ids_current_minibatch[5:10], settings, param, cache)
print mf.forest[0].counts
