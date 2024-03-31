import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

def get_data(): 
    for event in summary_iterator('logs/train/events.out.tfevents.train.original.v2'):
        for value in event.summary.value:
            print(value.simple_value)


def plot_data(train_acc_data, val_acc_data, train_loss_data, val_loss_data):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    ax[0].plot(train_acc_data['Step'], train_acc_data['Value'], label='Train')
    ax[0].plot(val_acc_data['Step'], val_acc_data['Value'], label='Validation')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    ax[1].plot(train_loss_data['Step'], train_loss_data['Value'], label='Train')
    ax[1].plot(val_loss_data['Step'], val_loss_data['Value'], label='Validation')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.show()

if __name__ == '__main__':
    get_data()