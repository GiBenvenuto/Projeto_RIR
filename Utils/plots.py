import csv
import numpy as np
import matplotlib.pyplot as plt


def salve_training_data(loss, name):
    csv.field_size_limit(393216)
    with open("Files/" + name, 'w') as csvfile:
        for i in loss:
            row = str(i) + "\n"
            csvfile.write(row)

    csvfile.close()


def load_training_data(name):
    file = open("Files/" + name)
    loss = csv.reader(file)
    loss = list(loss)
    loss = np.hstack(loss)
    loss = loss.astype(np.float)

    return loss


def plot_graph(name):
    y = load_training_data("trains-loss")
    x = range(len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.ylim(0.0, 1.0)
    plt.title(name)
    plt.xlabel("Épocas")
    plt.ylabel("NCC")
    fig.savefig("Files/" + name + ".png")
    plt.show()
