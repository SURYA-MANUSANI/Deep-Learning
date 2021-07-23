import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):
        """
        Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        :param x: Input Data
        :param y: Labels
        :return: loss value
        """
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model(x)
        # -calculate the loss
        loss = self._crit(output, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()



    def val_test_step(self, x, y):

        """
        Reference: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        :param x: Input Data
        :param y: Labels
        :return: loss value, predictions
        """
        # predict
        output = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(output, y)
        # return the loss and the predictions
        return loss.item(), output

    def train_epoch(self):
        # set training mode
        self._model.train()
        running_loss = 0.0
        total_data = 0
        # iterate through the training set
        for i, data_set in enumerate(self._train_dl, 0):
            data, label = data_set
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                data = data.cuda()
                label = label.cuda()
            # perform a training step
            loss = self.train_step(data, label)
            running_loss += loss
            total_data += len(data)
        # calculate the average loss for the epoch and return it
        average_loss = running_loss/total_data
        return average_loss

    def val_test(self):
        # set eval mode
        self._model.eval()
        predictions = []
        labels = []
        running_loss = 0.0
        total_data = 0
        # disable gradient computation
        with t.no_grad():
        # iterate through the validation set
            for i, data_set in enumerate(self._val_test_dl, 0):
                data, label = data_set
        # transfer the batch to the gpu if given
                if self._cuda:
                    data = data.cuda()
                    label = label.cuda()
        # perform a validation step
                loss, prediction = self.val_test_step(data, label)
        # save the predictions and the labels for each batch
                predictions.extend(prediction)
                labels.extend(label)
                running_loss += loss
                total_data += len(data)
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        average_loss = running_loss/total_data
        #metrics = self._calculate_metrics(predictions, labels)
        #average_f1 = metrics/total_data
       # print("\n\nAverage Validation split F1 score: {}\n\n".format(average_f1))
        # return the loss and print the calculated metrics
        return average_loss


    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_loss = []
        val_loss = []
        val_f1 = []
        epoch = 0
        while True:
            # stop by epoch number
            print(epoch,epochs)
            if epoch > epochs:
                break #return train_loss, val_loss, val_f1
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss_current = self.train_epoch()
            print(train_loss_current)
            val_loss_current= self.val_test()
            print(val_loss_current)
            # append the losses to the respective lists
            train_loss.append(train_loss_current)
            val_loss.append(val_loss_current)
            #val_f1.extend(val_f1_current)
            # use the save_checkpoint function to save the model for each epoch
            epoch += 1
            self.save_checkpoint(epoch)

            print(epoch)
            # check whether early stopping should be performed using the early stopping callback and stop if so
            # return the loss lists for both training and validation
            #if self._early_stopping_cb(val_loss_current):  # assert satisfied only if callback is set!
        return train_loss, val_loss                     #, val_f1
