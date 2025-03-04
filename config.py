############### Pytorch CIFAR configuration file ###############
import math

start_epoch = 1
num_epochs = 400
#batch_size = 256
#batch_size = 32
optim_type = 'SGD'

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'isic2019': (0.4914, 0.4822, 0.4465),
    'dogs_cats': (0.4914, 0.4822, 0.4465),
    'ba4': (0.4914, 0.4822, 0.4465),
    'siim': (0.4914, 0.4822, 0.4465),
    'imagenet': (0.4914, 0.4822, 0.4465),

}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'isic2019': (0.2023, 0.1994, 0.2010),
    'dogs_cats': (0.2023, 0.1994, 0.2010),
    'ba4': (0.2023, 0.1994, 0.2010),
    'siim': (0.2023, 0.1994, 0.2010),
    'imagenet': (0.2023, 0.1994, 0.2010),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# exp lr_schedule2
def learning_rate_mtd1(initial_lr, epoch):
    optim_factor = 0
    if (epoch > 160):
        optim_factor = 3
    elif (epoch > 120):
        optim_factor = 2
    elif (epoch > 60):
        optim_factor = 1

    return initial_lr * math.pow(0.2, optim_factor)


# exp lr_schedule1
def learning_rate_mtd2(initial_lr, epoch):
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


def learning_rate_mtd4(initial_lr, epoch, power_factor=1):
    if epoch < 10:
        initial_lr = initial_lr
    if epoch >= 10:
        initial_lr = 0.01*power_factor
    if epoch >20:
        initial_lr = 0.008*power_factor
    if epoch > 30:
        initial_lr = 0.001*power_factor
    if epoch > 40:
        initial_lr = 0.0008*power_factor
    if epoch > 50:
        initial_lr = 0.0004*power_factor
    if epoch > 60:
        initial_lr = 0.0001*power_factor
    if epoch > 70:
        initial_lr = 0.00004*power_factor
    if epoch > 80:
        initial_lr = 0.00001*power_factor
    if epoch > 90:
        initial_lr = 0.000008*power_factor
    return initial_lr


# default lr_schedule
# default lr_schedule
def learning_rate_mtd3(initial_lr, epoch):
    if epoch < 150:
        return initial_lr
    elif epoch < 250:
        return 0.01
    elif epoch < 350:
        return 0.001
    else:
        return 0.0001


def learning_rate_mtd5(initial_lr, epoch, power_factor=1):
    if epoch < 15:
        initial_lr = initial_lr
    if epoch >= 15:
        initial_lr = 0.01*power_factor
    if epoch >30:
        initial_lr = 0.008*power_factor
    if epoch > 45:
        initial_lr = 0.001*power_factor
    if epoch > 60:
        initial_lr = 0.0008*power_factor
    if epoch > 75:
        initial_lr = 0.0004*power_factor
    if epoch > 90:
        initial_lr = 0.0001*power_factor
    if epoch > 105:
        initial_lr = 0.00004*power_factor
    if epoch > 120:
        initial_lr = 0.00001*power_factor
    if epoch > 120:
        initial_lr = 0.000008*power_factor
    return initial_lr



def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s
