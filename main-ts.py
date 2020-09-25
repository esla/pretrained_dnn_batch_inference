from __future__ import print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim

import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import pandas as pd
import pickle
import csv

from networks import *
from torch.autograd import Variable

import torchvision.models as models
from efficientnet_pytorch import EfficientNet

from torch.utils.data import WeightedRandomSampler
#from torchsampler import ImbalancedDatasetSampler


from auglib.dataset_loader import FolderDatasetWithImgPath
from auglib.augmentation_pytorch.augmentations import Augmentations
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName

# Metrics
from model_evaluation.model_evaluator import ModelEvaluator
from model_evaluation.eval_utils import get_target_in_appropriate_format, temperature_scale

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

# esla introducing experimental loss functions
from experimental_dl_codes.from_kaggle_post_focal_loss import FocalLoss as FocalLoss1
from experimental_dl_codes.focal_loss2 import FocalLoss as FocalLoss2
from experimental_dl_codes.ohem import NllOhem
from experimental_dl_codes.focal_loss3 import FocalLoss as FocalLoss3
from experimental_dl_codes.other_transforms import ClaheTransform, RandomZeroPaddedSquareResizeTransform

# Config
import config as cf


def get_loss_criterion(args, gamma, alpha):
    global criterion
    # Loss functions
    if args.learning_type == 'multi_class':
        criterion = nn.CrossEntropyLoss()
    elif args.learning_type == 'multi_label':
        criterion = nn.BCEWithLogitsLoss()
    elif args.learning_type in ['focal_loss_target', 'focal_loss_ohe']:
        criterion = FocalLoss2(alpha=alpha, gamma=gamma)
    else:
        sys.exit('Unknown loss function type')
    return criterion


def get_network_32(args, num_classes):  # To Do: Currently works only for num_classes = 10
    if args.net_type == 'lenet':
        net = LeNet(num_classes)
        #net = LeNet()
        file_name = 'lenet'

    elif 'VGG' in args.net_type.upper():
        net = VGG(args.net_type.upper(), num_classes=num_classes)
        file_name = args.net_type
    elif args.net_type == 'resnet18':
        net = ResNet18(num_classes)
        file_name = args.net_type
    elif args.net_type == 'resnet34':
        net = ResNet34(num_classes)
        file_name = args.net_type
    elif args.net_type == 'resnet50':
        net = ResNet50(num_classes)
        file_name = args.net_type
    elif args.net_type == 'resnet101':
        net = ResNet101(num_classes)
        file_name = args.net_type
    elif args.net_type == 'resnet152':
        net = ResNet101(num_classes)
        file_name = args.net_type
    elif args.net_type == 'resnext29_2x64d':
        net = ResNeXt29_2x64d(num_classes=num_classes)
        file_name = 'resnext29_2x64d'
    elif args.net_type == 'efficientnetB0':
        net = EfficientNetB0(num_classes=num_classes)
        file_name = args.net_type
    # elif args.net_type == 'wide-resnet':
    #     net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
    #     file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_f/ctor)
    else:
        print('Error : Wrong Network selected for this input size')
        sys.exit(0)
    return net, file_name


# Return network and file name
def get_network_224(args, num_classes):
    if args.net_type == 'resnet18':
        file_name = args.net_type
        net = models.resnet18(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)

    elif args.net_type == 'resnet50':
        file_name = args.net_type
        net = models.resnet50(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)

    elif args.net_type == 'resnet101':
        file_name = args.net_type
        net = models.resnet101(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)

    elif args.net_type == 'wide_resnet50_2':
        file_name = args.net_type
        net = models.wide_resnet50_2(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)

    elif args.net_type == 'wide_resnet101_2':
        file_name = args.net_type
        net = models.wide_resnet101_2(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)

    elif args.net_type == 'resnext50_32x4d':
        file_name = args.net_type
        net = models.resnext50_32x4d(pretrained=True)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)

    elif args.net_type == 'resnext101_32x8d':
        file_name = args.net_type
        net = models.resnext101_32x8d(pretrained=False)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_classes)

    elif args.net_type == 'vgg11':
        file_name = args.net_type
        net = models.vgg11(pretrained=False)
        num_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_features, num_classes)
    elif args.net_type == 'vgg16':
        file_name = args.net_type
        net = models.vgg16(pretrained=True)
        num_features = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(num_features, num_classes)

    elif args.net_type == 'squeezenet1_1':
        file_name = args.net_type
        net = models.squeezenet1_1(pretrained=False)
        net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        net.num_classes = num_classes

    elif args.net_type == 'densenet121':
        file_name = args.net_type
        net = models.densenet121(pretrained=False)
        num_features = net.classifier.in_features
        net.classifier = nn.Linear(num_features, num_classes)

    else:
        print('Error : Wrong Network selected for this input image size')
        sys.exit(0)

    return net, file_name


def get_efficientnet_network(args, num_classes, pre_trained):
    """

    :param model_name: an example is 'efficientnet-b5'
    :return:
    """
    efficientnet_models = {
        'efficientnet-b0': 224,
        'efficientnet-b1': 240,
        'efficientnet-b2': 260,
        'efficientnet-b3': 300,
        'efficientnet-b4': 340,
        'efficientnet-b5': 456,
        'efficientnet-b6': 528,
        'efficientnet-b7': 600,
        'efficientnet-b8': 672
    }
    model_name = args.net_type
    file_name = args.net_type
    assert model_name in efficientnet_models.keys(), "EfficientNet model name is invalid!"
    assert args.input_image_size in efficientnet_models.values(), "Input_image_size is in-valid!"
    if not pre_trained:
        net = EfficientNet.from_name(model_name)
        net._fc = nn.Linear(in_features=net._fc.in_features, out_features=num_classes, bias=True)
    else:
        net = EfficientNet.from_pretrained(model_name)
        net._fc = nn.Linear(in_features=net._fc.in_features, out_features=num_classes, bias=True)

    required_input_image_size = EfficientNet.get_image_size(model_name)
    assert required_input_image_size == args.input_image_size, "input_image_size must matched model's required input size"
    return net, file_name


class MelanomaEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0', num_classes=2, pool_type=F.adaptive_avg_pool2d):
        super().__init__()
        self.pool_type = pool_type
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_features = getattr(self.backbone, '_fc').in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.pool_type(self.backbone.extract_features(x), 1)
        features = features.view(x.size(0),-1)
        #print(features.shape)
        return self.classifier(features)


#def get_model(model_name='efficientnet-b0', lr=1e-5, wd=0.01, freeze_backbone=False, opt_fn=torch.optim.AdamW,
#              device=None):
def get_model(args, num_classes, freeze_backbone=False):
    #device = device if device else get_device()
    model = MelanomaEfficientNet(model_name=args.net_type,  num_classes=num_classes)
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    #opt = opt_fn(model.parameters(), lr=lr, weight_decay=wd)
    if use_cuda:
        #model = model.to(device)
        model = model.cuda()
    return model, args.net_type


def get_network(args, num_classes):
    # generate list for efficientnet models
    efficientnet_model_names = ["efficientnet-b" + str(i) for i in range(9)]

    if args.net_type in efficientnet_model_names:
        #return get_efficientnet_network(args, num_classes, pre_trained=False)
        return get_model(args, num_classes)

    if args.input_image_size == 32:
        return get_network_32(args, num_classes)
    elif args.input_image_size == 224:
        return get_network_224(args, num_classes)
    else:
        sys.exit("!!!! Unknown args.input_image_size !!!!")


def get_learning_rate(args, epoch):
    if args.lr_scheduler == 'mtd1':
        return cf.learning_rate_mtd1(args.lr, epoch)
    elif args.lr_scheduler == 'mtd2':
        return cf.learning_rate_mtd2(args.lr, epoch)
    elif args.lr_scheduler == 'mtd3':
        return cf.learning_rate_mtd3(args.lr, epoch)
    elif args.lr_scheduler == 'mtd4':
        return cf.learning_rate_mtd4(args.lr, epoch, 0.1)
    elif args.lr_scheduler == 'mtd5':
        return cf.learning_rate_mtd5(args.lr, epoch, 0.1)
    else:
        sys.exit("Error! Unrecognized learing rate scheduler")


def get_class_distribution(dataset):
    dist_count = dict(Counter(dataset.targets))
    return dist_count


def get_class_weights(dataset):
    target_list = torch.tensor(dataset.targets)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in get_class_distribution(dataset).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    # Assign the weight of each class to all the samples
    class_weights_all = class_weights[target_list]
    #print(class_weights_all)
    return class_weights_all


def show_dataloader_images(data_loader, augs, idx, is_save=False, save_dir="./sample_images"):
    # Get a batch of training data
    (inputs, targets), img_names = next(iter(data_loader))
    #print(inputs.size())
    #print(img_names)
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    full_img_name = os.path.join(save_dir, str(idx)+".png")
    show_images(out, full_img_name, augs, is_save=True, title=[x for x in img_names])


def show_images(inp, full_img_name, augs, is_save, title):
    inp = inp.numpy().transpose((1, 2, 0))
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    mean, std = augs.mean, augs.std
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated
    if is_save:
        print("saving {}".format(full_img_name))
        plt.savefig(full_img_name)
    else:
        plt.show()


def train_model(net, optimizer, scheduler, epoch, args):
    global last_saved_lr, load_from_last_best, current_lr
    net.train()
    net.training = True
    train_loss = 0
    total = 0
    correct = 0
    # lr = cf.learning_rate(args.lr, epoch)
    #lr = args.lr

    # metrics
    metrics = {}


    # optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(lr, epoch))

    # print out current state of the learning rate
    print("\n\n Current LR: ", optimizer.param_groups[0]['lr'])
    current_lr = optimizer.param_groups[0]['lr']

    print('\n=> Training Epoch #%d' % epoch)
    for batch_idx, data in enumerate(train_loader):
        (inputs, targets), img_names = data
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # Forward Propagation
        pred_probs, pred_labels = torch.max(outputs.data, 1)

        loss = criterion(outputs, get_target_in_appropriate_format(args, targets, num_classes))

        # esla Experimental Idea: updating loss based on NLL for the incorrectly classified for every batch
        # loss_corr, loss_incorr, _, _ = decompose_loss(outputs, targets, pred_labels, criterion)

        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        if args.lr_scheduling_mtd != 'custom':
            scheduler.step()

        train_loss += loss.item()
        total += targets.size(0)
        correct += pred_labels.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')

        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] \t\t Loss: %.4f Acc@1: %.3f%% '
                         % (epoch, num_epochs, batch_idx + 1,
                        (len(train_set) // batch_size) + 1, loss.item(), 100. * correct / total))

        sys.stdout.flush()

    print('|\t\tLoss: %.4f Acc@1: %.3f%%' % (train_loss / batch_idx, 100. * correct / total))
    # metrics
    accuracy = 100. * correct / total
    metrics['accuracy'] = accuracy.cpu().data.numpy()
    metrics['train_loss'] = train_loss / batch_idx

    return metrics


def test_model(net, dataset_loader, class_dict, epoch=None, is_validation_mode=False):
    global best_accuracy, best_balanced_accuracy, best_acc_ace, logits, true_labels, pred_labels
    global save_point

    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    evaluator = ModelEvaluator(criterion, class_dict, args, with_image_names=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_loader):
            (inputs, targets), img_names = data
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            # Do temperature scaling here
            outputs = temperature_scale(outputs, 1.25)

            if is_validation_mode:
                # loss = criterion(outputs, get_one_hot_embedding(targets, num_classes))  # Loss for multi-label loss
                loss = criterion(outputs, get_target_in_appropriate_format(args, targets, num_classes))
                test_loss += loss.item()

            softmax_scores = F.softmax(outputs, dim=1)
            pred_probs, pred_labels = torch.max(softmax_scores.data, 1)
            total += targets.size(0)
            correct += pred_labels.eq(targets.data).cpu().sum()

            # Update Evaluator instance for evaluations
            evaluator.update_results(outputs, targets, img_names)

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] ' % (batch_idx, len(dataset_loader)))
            sys.stdout.flush()

    # Compute various evaluation metrics
    df_raw_vals, metrics, per_class_accs = evaluator.compute_all_metrics
    print(metrics)

    loss_corr = metrics['test_loss_corrects']
    loss_incorr = metrics['test_loss_incorrects']
    ece_total = metrics['ece_total']
    ece_pos = metrics['ece_pos_gap']
    ece_neg = metrics['ece_neg_gap']
    accuracy = metrics['accuracy']
    balanced_accuracy = metrics['balanced_accuracy']
    auc = metrics['auc']

    if is_validation_mode:
        loss = test_loss / batch_idx
        # loss_corr = loss_correctly_preds.item()
        # loss_incorr = loss_incorrectly_preds.item()
        print('-' * 60)
        print(f"| Val. Epoch #{epoch}")
        print('-' * 60)
        print(f"| Loss: {loss:.4f}  |  Corr Loss: {loss_corr:.4f}  |  Incorr Loss: {loss_incorr:.4f}")
        print(f"| Acc: {accuracy:.2f} | Bal. Acc: {balanced_accuracy:.2f} | AUC: {auc:.2f} |")
        print(f"| ECE Total: {ece_total:.4f}  |  ECE Pos: {ece_pos:.4f}  |  ECE Neg: {ece_neg:.4f}\n")

    else:
        print("\n| \t\t\t Acc@1: %.2f%% | BalAcc@1: %.2f%% | ECE: %.6f | auc: %.2f%%" %
              (accuracy, balanced_accuracy, ece_total, auc))

    if is_validation_mode and not args.inference_only:
        # if accuracy > best_accuracy  or accuracy  - metrics['test_loss_incorrects'] > best_acc_ace:  # stopping criteria idea 2
        if (balanced_accuracy > best_balanced_accuracy or accuracy > best_accuracy or accuracy -
                metrics['test_loss_incorrects'] > best_acc_ace):  # stopping criteria idea 2

            if accuracy > best_accuracy:
                suffix = 'val_acc'
            elif balanced_accuracy > best_balanced_accuracy:
                suffix = 'bal_acc'
            else:
                suffix = 'val_acc_incorr_nll'

            print('| Saving Best model and results...\t\t\tTop1 = %.2f%%' % accuracy)
            state = {
                'whole_model': net,
                'model': net.module if use_cuda else net,
                'acc': accuracy,
                'epoch': epoch,
            }

            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            # saved_model_overall_best = save_point + file_name + '.pth'
            saved_model_curr_best = save_point + file_name + '-epoch-' + str(epoch) + '.pth'
            # print("Saving model: {}".format(saved_model_curr_best))
            torch.save(state, saved_model_curr_best)
            # print("Saving model: {}".format(saved_model_overall_best))
            # torch.save(state, saved_model_overall_best)
            val_csv_file = save_point + file_name + '-epoch-' + str(epoch) + '-' + suffix + '.csv'
            # print("saving validation results: {}".format(val_csv_file))
            df_raw_vals.to_csv(val_csv_file)

            best_accuracy = accuracy  # default stopping criteria idea
            # best_accuracy = (accuracy/100) - metrics['ece_total']  # stopping criteria idea 1
            best_acc_ace = accuracy - metrics['test_loss_incorrects']  # stopping criteria idea 2
            best_balanced_accuracy = balanced_accuracy

    return df_raw_vals, metrics


def get_focal_loss_parameters():
    if args.alpha in ['None', '0.0', '0']:
        alpha = None
    else:
        alpha = float(args.alpha) + (val_metrics['balanced_accuracy'] - 50.0) / 1000  # focal loss improvement idea 1
    if args.train_loss_idea == 'ce':
        gamma = 0.0
    elif args.train_loss_idea == 'loss_idea1':
        gamma = 10 * val_metrics['ece_pos_gap']  # loss idea 1
    elif args.train_loss_idea == 'loss_idea2':
        gamma = val_metrics['test_loss_incorrects']  # loss idea 2
    elif args.train_loss_idea == 'loss_idea3':
        gamma = val_metrics['test_loss_incorrects'] + 10 * val_metrics['ece_total']  # loss idea 3
    elif args.train_loss_idea == 'loss_idea4':
        gamma = val_metrics['test_loss_incorrects'] + 10 * val_metrics['ece_pos_gap']  # loss idea 4
    else:
        sys.exit('Error!, Choose a valid training loss idea')
    return alpha, gamma


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Image Classification')

    task_selection_group = parser.add_argument_group('learning task type')
    dataset_params_group = parser.add_argument_group("dataset params")
    training_group = parser.add_argument_group("training params")
    inference_only_group = parser.add_argument_group("test-only params")

    # task type arguments
    task_selection_group.add_argument('--learning_type', default='multi_class', type=str, help=""" to select the kind of
                                                                                               learning""")
    # dataset parameters group arguments
    dataset_params_group.add_argument('--input_image_size', type=int, help='input image size for the network')
    dataset_params_group.add_argument('--dataset_class_type', '-dct', help='The class type for the dataset')
    dataset_params_group.add_argument('--datasets_class_folders_root_dir', '-folders_dir', help='Root dir for all dataset')
    dataset_params_group.add_argument('--datasets_csv_root_dir', '-csv_dir', help='Root dir for all dataset csv files')
    dataset_params_group.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    dataset_params_group.add_argument('--data_transform', type=str, help='The dataset transform to be used')


    # Ideas that I am exploring
    training_group.add_argument('--train_loss_idea', '-tl', type=str, help='Select the training loss idea to use')
    training_group.add_argument('--temp_scale_idea', '-ts', type=str, help='Select the temperature scaling to use')

    training_group.add_argument('--validate_train_dataset', '-t', action='store_true', help='resume from checkpoint')
    training_group.add_argument('--resume_training', '-r', action='store_true', help='resume from checkpoint')
    training_group.add_argument('--estimate_lr', '-lre',action='store_true', help='Use LR Finder to get rough '
                                                                                  'estimate of start lr')
    training_group.add_argument('--resume_from_model', '-rm', help='Model to load to resume training from')
    training_group.add_argument('--lr', default=0.001, type=float, help='learning_rate')
    training_group.add_argument('--alpha', '-a', default=None, type=str, help='alpha value for focal loss')
    training_group.add_argument('--lr_scheduling_mtd', type=str, help='choose the lr scheduling mechanism')
    training_group.add_argument('--lr_scheduler', type=str, help='Select the LR scheduler')
    training_group.add_argument('--batch_size', default=32, type=int, help='training batch size')
    training_group.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    training_group.add_argument('--depth', default=28, type=int, help='depth of model')
    training_group.add_argument('--widen_factor', default=10, type=int, help='width of model')
    training_group.add_argument('--dropout', default=0.0, type=float, help='dropout_rate')

    # inference related arguments
    inference_only_group.add_argument('--inference_only', '-i', action='store_true',
                                      help='Make inference mode with the saved model')
    inference_only_group.add_argument('--inference_model', '-im', help='Model to load for inference')
    inference_only_group.add_argument("--inference_dataset_dir", "-idir",
                                      help="root directory for inference class folders or CSV files")
    inference_only_group.add_argument('--inference_filename', '-ifn', type=str, help='file name to save inference '
                                                                                     'results')

    args = parser.parse_args()

    # Ensure the datasets root directory is valid
    if args.dataset not in ['cifar10', 'cifar100', 'ba4_project']:
        assert os.path.isdir(args.datasets_class_folders_root_dir), 'Please provide a valid root directory for all ' \
                                                                    'datasets'

    datasets_root_dir = args.datasets_class_folders_root_dir
    csv_root_dir = args.datasets_csv_root_dir

    dataset_class_type = args.dataset_class_type

    is_training = not args.inference_only or args.resume_training
    is_inference = args.inference_only

    # set priority and ensure that only one option is available at any time
    is_inference = False if is_training else is_inference

    # initialize needed variables to None
    train_root = None
    inference_root = None
    train_set = None
    val_set = None
    inference_set = None
    train_loader = None
    val_loader = None
    inference_loader = None

    # Update batch size if batch_size is provided as argument else use standard 32
    batch_size = args.batch_size

    # Hyper Parameter settings
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    best_accuracy = 0
    best_balanced_accuracy = 0

    start_epoch, num_epochs, optim_type = cf.start_epoch, cf.num_epochs, cf.optim_type

    # Optimizers
    # optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    # misc variables
    experiment_dir = datetime.now().strftime("%d-%b-%Y-%H_%M_%S.%f")[:-3]
    save_point = ''

    # Transformations
    aug = dict(hflip=False, vflip=False, rotation=0, shear=0, scale=1.0, color_contrast=0, color_saturation=0,
               color_brightness=0, color_hue=0, random_crop=False, random_erasing=False, piecewise_affine=False,
               tps=False, autoaugment=False)

    aug['size'] = args.input_image_size
    aug['mean'], aug['std'] = cf.mean[args.dataset], cf.std[args.dataset]

    augs = Augmentations(**aug)
    
    # Temporary transformations (temp trans1)
    # Data Uplaod
    # data_transform1
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(augs.size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(augs.mean, augs.std),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(augs.mean, augs.std),
    ])

    # Data augmentation and normalization for training (temp trans2)
    # Just normalization for validation
    # esla TO DO: data transformation should contain no hardcoded values
    # data_transform2
    data_transforms2 = {
        'train': transforms.Compose([
            #transforms.Resize(augs.size),
            transforms.RandomResizedCrop(augs.size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(augs.size),
            transforms.CenterCrop(augs.size),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
    }
    
    data_transforms3 = {
        'train': transforms.Compose([
            transforms.Resize(augs.size),
            transforms.RandomResizedCrop(augs.size),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(augs.size),
            transforms.CenterCrop(augs.size),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
    }

    data_transforms4 = {
        'train': transforms.Compose([
            # ClaheTransform(),
            # transforms.ToPILImage(),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.Resize(augs.size),
            transforms.RandomResizedCrop(augs.size, scale=(0.8, 1.0)),
            torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
        'val': transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            # ClaheTransform(),
            # transforms.ToPILImage(),
            transforms.Resize(augs.size, augs.size),
            # transforms.CenterCrop(augs.size),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
    }

    data_transforms5 = {
        'train': transforms.Compose([
            ClaheTransform(),
            transforms.ToPILImage(),
            #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
            #transforms.Grayscale(num_output_channels=3),
            transforms.RandomGrayscale(p=0.5),
            # transforms.Resize(augs.size),
            #RandomZeroPaddedSquareResizeTransform(square_crop_size=224), # my augmentation idea 1
            #transforms.CenterCrop(512),
            transforms.RandomResizedCrop(augs.size, scale=(0.8, 1.0)),
            #transforms.ColorJitter(brightness=(0.02, 2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
        'val': transforms.Compose([
            #transforms.Grayscale(num_output_channels=3),
            ClaheTransform(),
            transforms.ToPILImage(),
            #transforms.CenterCrop(512),
            transforms.Resize((augs.size, augs.size)),
            #transforms.CenterCrop(augs.size),
            transforms.RandomResizedCrop(augs.size, scale=(0.9, 1.0)),
            #RandomZeroPaddedSquareResizeTransform(square_crop_size=224),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
    }

    # Select the appropriate data transformation
    if args.data_transform == 'data_transform1':
        train_transform = transform_train
        val_transform = transform_test
    elif args.data_transform == 'data_transform2':
        train_transform = data_transforms2['train']
        val_transform = data_transforms2['val']
    elif args.data_transform == 'data_transform3':
        train_transform = data_transforms3['train']
        val_transform = data_transforms3['val']
    elif args.data_transform == 'data_transform4':
        train_transform = data_transforms4['train']
        val_transform = data_transforms3['val']
    elif args.data_transform == 'data_transform5':
        train_transform = data_transforms5['train']
        val_transform = data_transforms5['val']
    else:
        sys.exit("Error! Please provide the appropriate transform")

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

    if is_training:

        if args.dataset == 'cifar10_orig':
            if is_training:
                print("| Preparing CIFAR-10 dataset...")
                sys.stdout.write("| ")
                train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=augs.no_augmentation)
                val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                       transform=augs.no_augmentation)
            aug['size'] = 32
            assert train_set and val_set is not None, "Please ensure that you have valid train and val dataset formats"

        elif args.dataset == 'cifar100_orig':
            print("| Preparing CIFAR-100 dataset...")
            sys.stdout.write("| ")
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                          transform=augs.tf_transform)
            val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                        transform=augs.no_augmentation)
            aug['size'] = 32
            assert train_set and val_set is not None, "Please ensure that you have valid train and val dataset formats"

        elif dataset_class_type == "class_folders":
            train_root = datasets_root_dir + "/train"
            val_root = datasets_root_dir + "/val"
            train_set = FolderDatasetWithImgPath(train_root, transform=train_transform)
            val_set = FolderDatasetWithImgPath(val_root, transform=val_transform)
            print("class info: {}".format(train_set.class_to_idx))

        elif dataset_class_type == "csv_files_":
            train_root = datasets_root_dir + "/train"
            val_root = datasets_root_dir + "/val"
            train_csv = csv_root_dir + "/" + args.dataset + "_train.csv"
            val_csv = csv_root_dir + "/" + args.dataset + "_val.csv"
            train_set = CSVDataset(root=train_root, csv_file=train_csv, image_field='image_path', target_field='NV',
                                   transform=augs.train_transform)
            val_set = CSVDatasetWithName(root=val_root, csv_file=val_csv, image_field='image_path', target_field='NV',
                                         transform=val_transform)
        # elif dataset_class_type == "csv_files_2":
        #
        #     train_dataset = PlantDataset(data=train_data, transforms=transforms["train_transforms"],
        #                                 soft_labels_filename=None
        #     )
        #     val_dataset = PlantDataset(data=val_data, transforms=transforms["val_transforms"],
        #         soft_labels_filename=None
        #     )
        else:
            sys.exit("Should never be reached!!! Check dtaset_class_type argument")

        # Get the data loaders for training
        num_classes = len(train_set.class_to_idx)
        class_dict = train_set.class_to_idx.keys()
        #train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=8,
        #                                           sampler=ImbalancedDatasetSampler(train_set))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                       num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    elif is_inference:
        inference_root = args.inference_dataset_dir
        if dataset_class_type == "class_folders":
            inference_set = FolderDatasetWithImgPath(inference_root, transform=val_transform)
        elif dataset_class_type == "csv files":
            inference_csv = inference_root + "/" + "isic2019_val.csv"
            inference_set = CSVDatasetWithName(root=inference_root, csv_file=inference_csv, image_field='image_path',
                                               target_field='NV', transform=val_transform)
        else:
            sys.exit("Should never be reached!!! Check dtaset_class_type argument")
        # ensure the inference format matches that of training.
        # To Do: In the case of inference for unlabeled dataset, one has to present the dataset in the same
        # format as for the training and validation. This needs to be improved
        assert len(inference_set.class_to_idx) > 1, """Current implementation requires inference data to be in the same 
                                                        directory structure as the test and val datasets"""
        num_classes = len(inference_set.class_to_idx)
        class_dict = inference_set.class_to_idx.keys()
        assert inference_set is not None, "Please ensure that you have valid inference dataset formats"
        inference_loader = torch.utils.data.DataLoader(inference_set, batch_size=batch_size, shuffle=False,
                                                       num_workers=4)

    else:
        sys.exit("ERROR! This place should never be reached")

    # Save experiment configurations
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    exp_name = "_".join([experiment_dir, args.net_type, args.dataset, str(args.input_image_size)])
    save_point = './checkpoint/' + exp_name + os.sep

    if not os.path.isdir(save_point) and is_training:
        os.makedirs(save_point)

        exp_conf_file = os.path.join(save_point, "training_settings.txt")
        with open(exp_conf_file, 'a+') as f:
            f.write('\n'.join(sys.argv[1:]))
    
    criterion = get_loss_criterion(args, gamma=0, alpha=eval(args.alpha))

    if args.inference_only:
        print('\n[Inference Phase] : Model setup')

        # For TS Model
        # results_file_name = "temperature_scaling_res"
        # net_name = "densenet"
        # result_filename = "pre-trained-nets"
        #
        # from external_libs.temperature_scaling.models import DenseNet
        # growth_rate = 12
        # depth = 40
        #
        # # Get densenet configuration
        # if (depth - 4) % 3:
        #     raise Exception('Invalid depth')
        # block_config = [(depth - 4) // 6 for _ in range(3)]
        #
        # model = DenseNet(
        #     growth_rate=growth_rate,
        #     block_config=block_config,
        #     num_classes=10
        # )
        #
        # # Load model state dict
        # save = "./external_libs/temperature_scaling/results"
        # model_filename = os.path.join(save, 'model.pth')
        # if not os.path.exists(model_filename):
        #     raise RuntimeError('Cannot find file %s to load' % model_filename)
        # state_dict = torch.load(model_filename)
        #
        # # Load validation indices
        # valid_indices_filename = os.path.join(save, 'valid_indices.pth')
        # if not os.path.exists(valid_indices_filename):
        #     raise RuntimeError('Cannot find file %s to load' % valid_indices_filename)
        # valid_indices = torch.load(valid_indices_filename)
        #
        # model.load_state_dict(state_dict)
        # # Wrap model if multiple gpus
        # if torch.cuda.device_count() > 1:
        #     net = torch.nn.DataParallel(model).cuda()
        # else:
        #     net = model.cuda()
        #print(net)



        checkpoint_file = args.inference_model
        assert os.path.exists(checkpoint_file) and os.path.isfile(
            checkpoint_file), 'Error: No checkpoint directory found!'
        net_name = os.path.basename(checkpoint_file)
        result_filename = os.path.dirname(checkpoint_file).split('/')[-1] + '_' + net_name
        os.chdir("./external_libs/temperature_scaling")
        checkpoint = torch.load(checkpoint_file)
        net = checkpoint['model']
        #net = torch.load(checkpoint_file)
        #
        #
        #
        # if use_cuda:
        #     net.cuda()
        #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        #     cudnn.benchmark = True

        print('\n=> Inference in Progress')
        if args.validate_train_dataset:
            all_results_df, metrics = test_model(net, inference_loader, class_dict, is_validation_mode=True)
        else:
            all_results_df, metrics = test_model(net, inference_loader, class_dict)

        logits = all_results_df['Logits']
        true_labels = all_results_df['TrueLabels']
        pred_labels = all_results_df['PredictedLabels']

        # write results
        dataset_category = 'inference'
        experiment_dir = result_filename
        if os.path.exists("inference_results" + "/" + experiment_dir):
            pass
        else:
            os.makedirs("inference_results" + "/" + experiment_dir)
        if args.validate_train_dataset:
            filename = "checkpoint" + "/" + experiment_dir + "/" + "inference_train_dataset.csv"
            prefix_result_file = args.dataset + "-" + str(
                args.input_image_size) + '_' + dataset_category + '_' + net_name + 'validated_train_dataset'
        else:
            filename = "inference_results" + "/" + experiment_dir + "/" + "inference.csv"

        #os.makedirs("inference_results" + "/" + experiment_dir )
        if args.validate_train_dataset and not args.inference_only:
            filename = "checkpoint" + "/" + experiment_dir + "/" + "inference_train_dataset.csv"
            prefix_result_file = args.dataset + "-" + str(args.input_image_size) + '_' + dataset_category + '_' + net_name + 'validated_train_dataset'
        else:
            filename = "inference_results" + "/" + experiment_dir + "/" + "inference.csv"
            prefix_result_file = experiment_dir
            print("On {}".format(filename))

        with open(filename, 'a+') as infile:
            csv_writer = csv.writer(infile, dialect='excel')
            csv_writer.writerow(list(metrics.values()))

        # Save results to files
        all_results_df.to_csv(filename)

        if args.validate_train_dataset:
            with open(prefix_result_file + '.logits', 'wb') as f:
                pickle.dump((true_labels, pred_labels, logits), f)

        sys.exit(0)

    # To quickly visualize the effect of transforms from the dataloader
    # for i in range(50):
    #    show_dataloader_images(train_loader, augs, i, is_save=False, save_dir="./sample_images")

    # Model
    print('\n[Phase 2] : Model setup')
    if args.resume_training:
        # Load checkpoint
        print('| Resuming from checkpoint...')
        _, file_name = get_network(args, num_classes)
        checkpoint_file = args.resume_from_model
        assert os.path.exists(checkpoint_file) and os.path.isfile(
            checkpoint_file), 'Error: No checkpoint directory found!'
        net_name = os.path.basename(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        net = checkpoint['model']
        start_epoch = 2
        best_accuracy = 5.0
    else:
        print('| Building net type [' + args.net_type + ']...')
        net, file_name = get_network(args, num_classes)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # get the appropriate loss criterion for training
    #criterion = get_loss_criterion(args, gamma=0, alpha=args.alpha)

    #steps = 10
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0, weight_decay=5e-4)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.000001, max_lr=0.01,step_size_up=10000, step_size_down=None, mode='triangular')


    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()

        # perform one epoch of training
        train_metrics = train_model(net, optimizer, scheduler, epoch, args)

        # validate_model(net, epoch)
        df_raw_val,  val_metrics = test_model(net, val_loader, class_dict, epoch, is_validation_mode=True)
        logits = df_raw_val['Logits']
        true_labels = df_raw_val['TrueLabels']
        pred_labels = df_raw_val['PredictedLabels']

        alpha, gamma = get_focal_loss_parameters()

        print(f"\n Next alpha value: {alpha}")
        print(f" Next gamma value: {gamma:.4f}")

        criterion = get_loss_criterion(args, gamma=gamma, alpha=alpha)

        if args.lr_scheduling_mtd == 'custom':
            optimizer = optim.SGD(net.parameters(), lr=get_learning_rate(args, epoch), momentum=0.9, weight_decay=5e-4)
        else:
            scheduler.step()

        # write results
        filename = os.path.join(save_point, "training_log.csv")
        with open(filename, 'a+') as infile:
            csv_writer = csv.writer(infile, dialect='excel')
            if epoch == 1:
                csv_writer.writerow([epoch] + list(train_metrics.keys()) + list(val_metrics.keys()))
            csv_writer.writerow([epoch] + list(train_metrics.values()) + list(val_metrics.values()))

        filename = save_point + args.net_type + '-' + str(epoch) + '-' + 'val'

        with open(filename + '.logits', 'wb') as f:
            pickle.dump((true_labels, pred_labels, logits), f)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))
