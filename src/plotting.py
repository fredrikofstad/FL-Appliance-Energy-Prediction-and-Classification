import matplotlib.pyplot as plt
import tikzplotlib

output = "../report/figures/"


def make_plot(title, loss):

    plt.style.use("ggplot")

    plt.plot(loss)
    plt.title('Training Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    tikzplotlib.save(output+title+".tex")


def make_comparison_plot(title, pred, target):
    plt.style.use("ggplot")
    plt.plot(target, label='Ground Truth')
    plt.plot(pred, label='Predicted Values')
    plt.title('Comparison of Predictions and Ground Truth')
    plt.xlabel('Day')
    plt.ylabel('Energy Consumption')
    plt.legend()

    tikzplotlib.save(output+title+".tex")
