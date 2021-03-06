import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from model import ResNet as res

batch = 30

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
dt = pd.read_csv('./data.csv', sep=';')

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
training_data, testing_data = train_test_split(dt, test_size = 0.15)
train_data_loading = t.utils.data.DataLoader(ChallengeDataset(training_data, 'train'), batch_size=batch, shuffle = True)
testing_data_load = t.utils.data.DataLoader(ChallengeDataset(training_data, 'test'), batch_size=batch, shuffle = False)


# create an instance of our ResNet model
model = res()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
lossfunc = t.nn.BCEWithLogitsLoss()
opt = t.optim.Adam(model.parameters(), lr = 0.00001)


# go, go, go... call fit on trainer
model_trainer = Trainer(model, lossfunc, opt, train_data_loading, testing_data_load, True)
res = model_trainer.fit(epochs= 200)


# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
