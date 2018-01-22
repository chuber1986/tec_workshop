import numpy as np
import sys

from tec.utils.plotting import DNNPlot
from tec.utils.torch_utils import predict_images, load_checkpoint


def predict(model_path, image_path):
    plt = DNNPlot(n_cols=5, n_rows=4)

    model, _, _ = load_checkpoint(model_path)
    print(model)

    plt.plot(model.fc1.weight.data)
    # plt.plot(model.conv1.weight.data)

    files, images, outputs = predict_images(model, image_path, threshhold=125)
    targets = [f[f.rfind('/') + 1: f.rfind('.')] for f in files]

    # plt.clear()
    plt.plot(images, ['Predicted: {}'.format(o) for o in outputs])

    plt = DNNPlot(n_cols=10, n_rows=10)
    for i in images:
        states = np.asarray(model.fc1.weight.data) * i.reshape(1, -1)
        plt.plot(np.asarray(states))

    for f, o, t in zip(files, outputs, targets):
        print('{} => Target: {} Output: {}'.format(f, t, o))

    print("Press Enter to continue...")
    sys.stdin.readline()


def main():
    model_path = '../../model/MNISTClassification1.pth.tar.gz'
    image_path = '../../images'

    predict(model_path, image_path)


if __name__ == '__main__':
    main()
