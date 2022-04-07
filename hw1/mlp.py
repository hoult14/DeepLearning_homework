import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

# input and output dimensions
input_dim = 784
output_dim = 10


def calc_loss_and_grad(x, y, w1, b1, w2, b2, eval_only=False):
    """Forward Propagation and Backward Propagation.

    Given a mini-batch of images x, associated labels y, and a set of parameters, compute the
    cross-entropy loss and gradients corresponding to these parameters.

    :param x: images of one mini-batch.  (batch_size, input_dim)
    :param y: labels of one mini-batch.  (batch_size, output_dim)
    :param w1: weight parameters of layer 1. (input_dim, l1_dim)
    :param b1: bias parameters of layer 1.   (1, l1_dim)
    :param w2: weight parameters of layer 2. (l1_dim, out_dim)
    :param b2: bias parameters of layer 2.   (1, out_dim)
    :param eval_only: if True, only return the loss and predictions of the MLP.
    :return: a tuple of (loss, db2, dw2, db1, dw1)
    """
    # TODO
    # forward pass
    #loss, y_hat = None, None
    batch_size = x.shape[0]
    z1 = np.matmul(x, w1) + b1
    h1 = np.maximum(z1, 0)
    z2 = np.matmul(h1, w2) + b2
    y_hat = np.exp(z2 - np.max(z2, axis = 1, keepdims=True))
    y_hat = y_hat / np.sum(y_hat, axis = 1, keepdims=True)
    loss = 1. / batch_size * np.sum(np.nan_to_num(-y * np.log(y_hat)))
    if eval_only:
        return loss, y_hat
    # TODO
    # backward pass
    #db2, dw2, db1, dw1 = None, None, None, None
    dY = (y_hat - y) / batch_size # [batch_size, out_dim]
    db2 = np.sum(dY, axis = 0, keepdims = True) # [1, out_dim]
    dw2 = np.matmul(h1.T, dY) # []
    #
    dY = np.matmul(dY, w2.T)  # [batch_size, l1_dim]
    dY = dY * (z1 > 0)
    db1 = np.sum(dY, axis = 0, keepdims = True) # [1, l1_dim]
    dw1 = np.matmul(x.T, dY)  

    return loss, db2.squeeze(), dw2, db1.squeeze(), dw1


def train(train_x, train_y, test_x, text_y, args: argparse.Namespace):
    """Train the network.

    :param train_x: images of the training set.
    :param train_y: labels of the training set.
    :param test_x: images of the test set.
    :param text_y: labels of the test set.
    :param args: a dict of hyper-parameters.
    """

    # TODO
    #  randomly initialize the parameters (weights and biases)
    # w1, b1, w2, b2 = None, None, None, None
    input_dim = train_x.shape[1]
    output_dim = train_y.shape[1]
    hidden_dim = args.hidden_dim
    w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim)
    w2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim)

    print('Start training:')
    print_freq = 100
    loss_curve = []

    for epoch in range(args.epochs):
        # 每50个迭代周期，学习率降低为原来的0.1倍
        if epoch > 0 and epoch % 20 == 0:
            args.lr = args.lr / 10
        # train for one epoch
        print("[Epoch #{}]".format(epoch))

        # random shuffle dataset
        dataset = np.hstack((train_x, train_y))
        np.random.shuffle(dataset)
        train_x = dataset[:, :input_dim]
        train_y = dataset[:, input_dim:]

        n_iterations = train_x.shape[0] // args.batch_size

        for i in range(n_iterations):
            # load a mini-batch
            x_batch = train_x[i * args.batch_size: (i + 1) * args.batch_size, :]
            y_batch = train_y[i * args.batch_size: (i + 1) * args.batch_size, :]

            # TODO
            # compute loss and gradients
            #loss = None
            loss, db2, dw2, db1, dw1 = calc_loss_and_grad(x_batch, y_batch, w1, b1, w2, b2)

            # TODO
            # update parameters
            w1 -= args.lr * dw1
            b1 -= args.lr * db1
            w2 -= args.lr * dw2
            b2 -= args.lr * db2

            loss_curve.append(loss)
            if i % print_freq == 0:
                print('[Iteration #{}/{}] [Loss #{:4f}]'.format(i, n_iterations, loss))

    # show learning curve
    plt.title('Training Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(loss_curve)), loss_curve)
    plt.show()

    # evaluate on the training set
    loss, y_hat = calc_loss_and_grad(train_x, train_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(train_y, axis=1)
    accuracy = np.sum(predictions == labels) / train_x.shape[0]
    print('Top-1 accuracy on the training set', accuracy)

    # evaluate on the test set
    loss, y_hat = calc_loss_and_grad(test_x, text_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(text_y, axis=1)
    accuracy = np.sum(predictions == labels) / test_x.shape[0]
    print('Top-1 accuracy on the test set', accuracy)


def main(args: argparse.Namespace):
    # print hyper-parameters
    print('Hyper-parameters:')
    print(args)

    # load training set and test set
    train_x, train_y = utils.load_data("train")
    test_x, text_y = utils.load_data("test")
    print('Dataset information:')
    print("training set size: {}".format(len(train_x)))
    print("test set size: {}".format(len(test_x)))
    print("{}".format(type(train_x)))

    # check your implementation of backward propagation before starting training
    utils.check_grad(calc_loss_and_grad)

    # train the network and report the accuracy on the training and the test set
    train(train_x, train_y, test_x, text_y, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multilayer Perceptron')
    parser.add_argument('--hidden-dim', default=50, type=int,
                        help='hidden dimension of the Multilayer Perceptron')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=60, type=int,
                        help='number of total epochs to run')
    args = parser.parse_args()
    main(args)
