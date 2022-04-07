import os
from os.path import dirname, exists
import numpy as np
import matplotlib.pyplot as plt


def load_data(split):
    """Load dataset.

    :param split: a string specifying the partition of the dataset ('train' or 'test').
    :return: a (images, labels) tuple of corresponding partition.
    """

    images = np.load("./data/mnist_{}_images.npy".format(split))
    labels = np.load("./data/mnist_{}_labels.npy".format(split))
    return images, labels


def check_grad(calc_loss_and_grad):
    """Check backward propagation implementation. This is naively implemented with finite difference method.
    You do **not** need to modify this function.
    """

    def relative_error(z1, z2):
        return np.mean((z1 - z2) ** 2 / (z1 ** 2 + z2 ** 2))

    print('Gradient check of backward propagation:')

    # generate random test data
    x = np.random.rand(5, 15)
    y = np.random.rand(5, 3)
    # construct one hot labels
    y = y * (y >= np.max(y, axis=1, keepdims=True)) / np.max(y, axis=1, keepdims=True)

    # generate random parameters
    w1 = np.random.rand(15, 3)
    b1 = np.random.rand(3)
    w2 = np.random.rand(3, 3)
    b2 = np.random.rand(3)

    # calculate grad by backward propagation
    loss, db2, dw2, db1, dw1 = calc_loss_and_grad(x, y, w1, b1, w2, b2)

    # calculate grad by finite difference
    epsilon = 1e-5

    numeric_dw2 = np.zeros_like(w2)
    for i in range(w2.shape[0]):
        for j in range(w2.shape[1]):
            w2[i, j] += epsilon
            loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
            w2[i, j] -= epsilon
            numeric_dw2[i, j] = (loss_prime - loss) / epsilon
    print('Relative error of dw2', relative_error(numeric_dw2, dw2))

    numeric_db2 = np.zeros_like(b2)
    for i in range(db2.shape[0]):
        b2[i] += epsilon
        loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
        b2[i] -= epsilon
        numeric_db2[i] = (loss_prime - loss) / epsilon
    print('Relative error of db2', relative_error(numeric_db2, db2))

    numeric_dw1 = np.zeros_like(w1)
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            w1[i, j] += epsilon
            loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
            w1[i, j] -= epsilon
            numeric_dw1[i, j] = (loss_prime - loss) / epsilon
    print('Relative error of dw1', relative_error(numeric_dw1, dw1))

    numeric_db1 = np.zeros_like(b1)
    for i in range(db1.shape[0]):
        b1[i] += epsilon
        loss_prime = calc_loss_and_grad(x, y, w1, b1, w2, b2)[0]
        b1[i] -= epsilon
        numeric_db1[i] = (loss_prime - loss) / epsilon

    print('Relative error of db1', relative_error(numeric_db1, db1))
    print('If you implement back propagation correctly, all these relative errors should be less than 1e-5.')


def save_figure(filename, show_figure=True):
    if not exists(dirname(filename)):
        os.makedirs(dirname(filename))
    plt.tight_layout()
    plt.savefig(filename)
    if show_figure:
        plt.show()


def save_scatter_2d(data, title, filename):
    plt.figure()
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1])
    save_figure(filename)


def plot_vae_training_plot(train_losses, test_losses, title, filename):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    save_figure(filename)


def sample_data_1_a(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_2_a(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + (rand.randn(count, 2) * [[1.0, 5.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])


def sample_data_1_b(count):
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]


def sample_data_2_b(count):
    rand = np.random.RandomState(0)
    return [[-1.0, 2.0]] + rand.randn(count, 2) * [[1.0, 5.0]]


def sample_data(part, dataset_id):
    assert dataset_id in [1, 2]
    assert part in ['a', 'b']
    if part == 'a':
        if dataset_id == 1:
            dataset_fn = sample_data_1_a
        else:
            dataset_fn = sample_data_2_a
    else:
        if dataset_id == 1:
            dataset_fn = sample_data_1_b
        else:
            dataset_fn = sample_data_2_b

    train_data, test_data = dataset_fn(10000), dataset_fn(2500)
    return train_data.astype('float32'), test_data.astype('float32')


def visualize_data(part, dataset_id):
    """Visualize corresponding datasets.
    """

    train_data, test_data = sample_data(part, dataset_id)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Train Data (Part {}, Dataset {})'.format(part, dataset_id))
    ax1.scatter(train_data[:, 0], train_data[:, 1])
    ax2.set_title('Test Data (Part {}, Dataset {})'.format(part, dataset_id))
    ax2.scatter(test_data[:, 0], test_data[:, 1])
    plt.show()


def save_results(part, dataset_id, train_and_sample_func, args):
    """Save results of VAE. You do **not** need to modify this function.

    :param part: a string that specifies how to construct samples.
    :param dataset_id: a string that specifies dataset.
    :param train_and_sample_func: a function that trains the VAE and generate samples with the trained model.
    :param args: a dict of hyper-parameters.
    """

    train_data, test_data = sample_data(part, dataset_id)
    train_losses, test_losses, samples_noise, samples_no_noise = train_and_sample_func(train_data, test_data, args)
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')

    plot_vae_training_plot(train_losses, test_losses, f'Part({part}) Dataset {dataset_id} Train Plot',
                           f'visualization/part_{part}_dataset_{dataset_id}_train_plot.png')
    save_scatter_2d(samples_noise, title='Samples with Decoder Noise',
                    filename=f'visualization/part_{part}_dataset_{dataset_id}_sample_with_noise.png')
    save_scatter_2d(samples_no_noise, title='Samples without Decoder Noise',
                    filename=f'visualization/part_{part}_dataset_{dataset_id}_sample_without_noise.png')
