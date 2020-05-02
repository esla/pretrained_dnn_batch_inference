############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
num_epochs = 3000
#batch_size = 256
#batch_size = 32
optim_type = 'SGD'

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'isic2019': (0.4914, 0.4822, 0.4465),
    'ba4': (0.4914, 0.4822, 0.4465),

}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'isic2019': (0.2023, 0.1994, 0.2010),
    'ba4': (0.2023, 0.1994, 0.2010),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def learning_rate_mtd1(initial_lr, epoch):
    optim_factor = 0
    if (epoch > 160):
        optim_factor = 3
    elif (epoch > 120):
        optim_factor = 2
    elif (epoch > 60):
        optim_factor = 1

    return initial_lr * math.pow(0.2, optim_factor)


def learning_rate_mine(initial_lr, epoch):
    if epoch < 20:
        initial_lr = initial_lr
    if epoch >= 20:
        initial_lr = 0.4
    if epoch >40:
        initial_lr = 0.1
    if epoch > 60:
        initial_lr = 0.04
    if epoch > 80:
        initial_lr = 0.01
    if epoch > 100:
        initial_lr = 0.008
    if epoch > 120:
        initial_lr = 0.004
    if epoch > 140:
        initial_lr = 0.001
    if epoch > 160:
        initial_lr = 0.0004
    if epoch > 180:
        initial_lr = 0.0001
    return initial_lr


def learning_rate(initial_lr, epoch):
    if epoch < 150:
        initial_lr = initial_lr
    elif epoch >= 150:
        initial_lr = 0.01
    elif epoch > 250:
        initial_lr = 0.001
    elif epoch > 350:
        initial_lr = 0.0001
    return initial_lr


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
