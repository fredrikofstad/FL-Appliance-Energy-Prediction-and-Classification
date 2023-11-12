import matplotlib.pyplot as plt
import tikzplotlib
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

output = "../report/figures/"


def make_loss_plot(title, loss):
    plt.style.use("ggplot")

    plt.plot(loss)
    plt.title('Training Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    tikzplotlib.save(output + title + ".tex")


def plot_pred_true(title, matrix):
    plt.style.use("ggplot")
    plt.plot(np.ravel(matrix[:, :, 1]), label='Ground Truth')
    plt.plot(np.ravel(matrix[:, :, 0]), label='Predicted Output')

    plt.title(f'Prediction vs Ground Truth')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.legend()
    tikzplotlib.save(output + title + ".tex")


def plot_pred_true_test(title, matrix):
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot y_pred as the first series
    ax.plot(np.ravel(matrix[:, :, 0]), label='Predicted Output')

    # Plot y_true as the second series
    ax.plot(np.ravel(matrix[:, :, 1]), label='Ground Truth')

    # Add labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')
    ax.legend()
    tikzplotlib.save(output + title + ".tex")


def make_accuracy_plot(title, accuracy):
    epoch_labels = [f'{i + 1}' for i in range(len(accuracy))]
    plt.style.use("ggplot")
    # Plot the accuracy values
    plt.plot(epoch_labels, accuracy, marker='o', linestyle='-', color='b')

    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')

    tikzplotlib.save(output + title + ".tex")


def make_comparison_plot(title, pred, target):
    plt.style.use("ggplot")
    plt.plot(target, label='Ground Truth')
    plt.plot(pred, label='Predicted Values')
    plt.title('Comparison of Predictions and Ground Truth')
    plt.xlabel('Day')
    plt.ylabel('Energy Consumption')
    plt.legend()

    tikzplotlib.save(output + title + ".tex")


def make_confusion_plot(title, confusion_list):
    # Get class labels
    class_labels = ['AC', 'Dishwasher', 'Washing Machine', 'Dryer', 'Water Heater',
                    'TV', 'Microwave', 'Kettle', 'Lighting', 'Refrigerator']
    # Plot each confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_list[50], annot=False, cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    tikzplotlib.save(output + title + ".tex")
