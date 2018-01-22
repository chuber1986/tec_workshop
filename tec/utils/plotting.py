from os.path import join
import matplotlib.pyplot as plt

from matplotlib import interactive
interactive(True)

import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap


class KMeansPlot:
    def __init__(self, t_mean_update=1, t_cluster_update=1):
        self.t_mean_update = t_mean_update
        self.t_cluster_update = t_cluster_update

        plt.title('K-Means')
        plt.show()

    def plot(self, x, means, iteration, is_mean_update=True):
        k = len(means)
        colors = cm.rainbow(np.linspace(0, 1, k))
        cmap = LinearSegmentedColormap.from_list('cmap', colors, N=k)

        plt.clf()

        mins = np.min(np.concatenate(x, axis=0), axis=0)
        maxs = np.max(np.concatenate(x, axis=0), axis=0)

        x_ = np.linspace(mins[0] - 0.5, maxs[0] + 0.5, 1000)
        y_ = np.linspace(mins[1] - 0.5, maxs[1] + 0.5, 1000)

        x_, y_ = np.meshgrid(x_, y_)
        grid = np.c_[np.reshape(x_, [-1]), np.reshape(y_, [-1])]

        diffs = []
        for mean in means:
            diffs.append(np.sqrt(np.sum(np.square(grid - mean), axis=1)))

        z_ = np.argmin(diffs, axis=0)
        _, idx = np.unique(z_, return_index=True)
        plt.contourf(x_, y_, z_.reshape(x_.shape), cmap=cmap, alpha=0.1)

        for i in range(k):
            plt.scatter(x[i][:, 0], x[i][:, 1], marker='.', color=colors[i], alpha=0.5)
        scatter = plt.scatter(means[:, 0], means[:, 1], marker='x', color='k')
        plt.text(-8, 8, 'Iteration: {}'.format(iteration + 1))

        scatter.figure.canvas.draw()

        if is_mean_update:
            plt.pause(self.t_mean_update)
        else:
            plt.pause(self.t_cluster_update)

    def save(self, path, fname):
        plt.savefig(join(path, fname + '.png'))


class TrainingPlot:
    def __init__(self, init_acc, init_loss):
        self.train_acc = []
        self.train_loss = []
        self.test_acc = [init_acc]
        self.test_loss = [init_loss]
        self.iteration = 0

        self.fig, ((self.ax1), (self.ax2)) = plt.subplots(2, 1, figsize=(12,8))
        self.fig.show()

    def plot(self, train_acc, train_loss, test_acc, test_loss):
        bs = len(train_loss)

        self.train_acc.extend(train_acc)
        self.train_loss.extend(train_loss)
        self.test_acc.append(test_acc)
        self.test_loss.append(test_loss)

        self.iteration += bs
        iteration = range(self.iteration + bs)

        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_title('Accuracy')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Accuracy')
        p1 = self.ax1.plot(iteration[:-bs], self.train_acc, '-r', label='Train accuracy')
        p1[0].figure.canvas.draw()
        p2 = self.ax1.plot(iteration[::bs], self.test_acc, '-b', label='Test accuracy')
        p2[0].figure.canvas.draw()
        self.ax1.legend()

        self.ax2.set_title('Loss')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Loss')
        p1 = self.ax2.plot(iteration[:-bs], self.train_loss, '-r', label='Train loss')
        p1[0].figure.canvas.draw()
        p2 = self.ax2.plot(iteration[::bs], self.test_loss, '-b', label='Test loss')
        p2[0].figure.canvas.draw()
        self.ax2.legend()

        plt.pause(0.5)

    def save(self, path, fname):
        self.fig.savefig(join(path, fname + '.png'))


class DNNPlot:
    def __init__(self, n_cols=1, n_rows=1):
        self.fig, self.axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        self.idx = 0

        if not isinstance(self.axes, np.ndarray):
            self.axes = np.asarray([self.axes]).reshape((1, 1))

        if len(self.axes.shape) < 2:
            self.axes = self.axes[:, None]

        self.clear()
        self.fig.show()

    def plot(self, imgs, labels=None):
        imgs = np.asarray(imgs)
        imgs = imgs.reshape(imgs.shape[0], -1)

        assert imgs.shape[0] <= np.prod(self.axes.shape) - self.idx
        assert labels is None or len(labels) == imgs.shape[0]

        a = np.abs(imgs).max()
        imgs[:, 0] = a
        imgs[:, 1] = -a

        for idx in range(imgs.shape[0]):
            ax = self.axes.flat[self.idx + idx]
            plot_image(ax, imgs[idx], labels[idx] if labels is not None else None)

        self.idx += imgs.shape[0]
        self.fig.canvas.draw()

        plt.pause(0.5)

    def save(self, path, fname):
        self.fig.savefig(join(path, fname + '.png'))

    def clear(self):
        self.idx = 0
        for ax in self.axes.flat:
            ax.cla()
            ax.axis('off')


def plot_image(ax, image, label=None):
    s = int(np.sqrt(image.shape[-1]))
    if image.min() < 0 or image.max() > 1.0:
        colormap = plt.cm.RdBu
    else:
        colormap = plt.cm.gray_r
    image = image.reshape(s, s)
    im = ax.imshow(image, interpolation='None', cmap=colormap)

    if label is not None:
        ax.text(2, 3, label,
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

    return im
