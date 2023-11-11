import matplotlib.pyplot as plt
import tikzplotlib

output = "../report/figures/"


def make_plot(title, loss):

    plt.style.use("ggplot")

    plt.plot(loss)
    plt.title('Training Loss per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')

    tikzplotlib.save(output+title+".tex")

