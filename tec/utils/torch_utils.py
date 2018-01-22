import os

import numpy as np
import torch
from os.path import isfile, isdir, join
import glob

from torch.autograd import Variable


def save_checkpoint(model_path, model, optimizer, epoch):
    model_path = join(model_path, model.__class__.__name__ + '.pth.tar.gz')
    torch.save({
            'epoch': epoch + 1,
            'model': model.__class__,
            'model_state': model.state_dict(),
            'optimizer': optimizer.__class__,
            'optimizer_state': optimizer.state_dict(),
        }, model_path)


def load_checkpoint(model_path):
    if not isfile(model_path):
        print("=> no checkpoint found at '{}'".format(model_path))

    print("=> loading checkpoint '{}'".format(model_path))

    checkpoint = torch.load(model_path)
    epoch = checkpoint['epoch']

    model = checkpoint['model']()
    model.load_state_dict(checkpoint['model_state'])

    optimizer = checkpoint['optimizer'](model.parameters(), lr=None)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, epoch))

    return model, optimizer, epoch


def predict_images(model, image_path, threshhold=150):
    try:
        from scipy.misc import imread, imresize, imfilter
    except ImportError:
        print("Couldn't import image library: scipy.")
        return

    files = []
    if isdir(image_path):
        for ending in ['jpg', 'jpeg', 'png']:
            files.extend(glob.glob(join(image_path, '*.' + ending)))
    elif isfile(image_path):
        files.append(image_path)

    imgs = []
    files = sorted(files)
    for f in files:
        print('Processing image: {}'.format(f))
        im = imread(f, flatten=True)
        im = imfilter(im, 'edge_enhance_more')
        h, w = im.shape
        im = im[:, int(w / 2 - h / 2):int(w / 2 + h / 2)]
        im = imresize(im, (28, 28))
        im = np.where(im > threshhold, 0, 1)

        imgs.append(im)

    files = np.asarray(files)
    imgs = np.asarray(np.stack(imgs))
    imgs = imgs.reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2])
    outputs = model(Variable(torch.from_numpy(imgs)).float())
    outputs = np.asarray(np.argmax(outputs.data, axis=1))

    return files, imgs, outputs
