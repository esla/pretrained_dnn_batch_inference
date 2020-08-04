from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


from auglib.dataset_loader import FolderDatasetWithImgPath
from auglib.augmentation import Augmentations
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName

# Metrics
from keras.utils.np_utils import to_categorical
from calibration.temp_api import get_adaptive_ece
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

# esla introducing experimental loss functions
from experimental_dl_codes.from_kaggle_post_focal_loss import FocalLoss as FocalLoss1
from experimental_dl_codes.focal_loss2 import FocalLoss as FocalLoss2
from experimental_dl_codes.ohem import NllOhem
from experimental_dl_codes.focal_loss3 import FocalLoss as FocalLoss3
from experimental_dl_codes.other_transforms import ClaheTransform

from torch_lr_finder import LRFinder

# Config
import config as cf


def get_topk_accuracy(outputs, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


def get_one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels].cuda()


def get_target_in_appropriate_format(args, targets, num_classes):
    if args.learning_type in ['multi_class', 'focal_loss_target']:
        return targets
    elif args.learning_type in ['multi_label', 'focal_loss_ohe']:
        return get_one_hot_embedding(targets, num_classes)
    else:
        sys.exit("Unknown learning task type")


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
    #     file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor)
    else:
        print('Error : Wrong Network selected for this input size')
        sys.exit(0)
    return net, file_name


def decompose_loss(logits, targets, predictions):
    # convert the type for targets to allow for concatenation with logits tesnsor
    targets = targets.type('torch.FloatTensor').view(len(targets), 1).cuda()
    predictions = predictions.type('torch.FloatTensor').view(len(predictions), 1).cuda()

    temp_result = torch.cat([logits, targets, predictions], 1)

    correct_cat = temp_result[:, :-1][temp_result[:, -1] == temp_result[:, -2]]
    incorrect_cat = temp_result[:, :-1][temp_result[:, -1] != temp_result[:, -2]]

    incorrect_outputs = incorrect_cat[:, :-1]
    # esla debugging
    # print("incorrect_cat[:, -1].view(1, len(incorrect_outputs))[0]",
    #       incorrect_cat[:, -1].view(1, len(incorrect_outputs))[0].shape)
    targets_for_incorrect = incorrect_cat[:, -1].view(1, len(incorrect_outputs))[0].type('torch.LongTensor')

    correct_outputs = correct_cat[:, :-1]
    # # esla debugging
    # print("correct_cat[:, -1].view(1, len(incorrect_outputs))[0]",
    #       correct_cat[:, -1].view(1, len(correct_outputs))[0].shape)

    # convert the type for targets back to torch.LongTensor
    targets_for_correct = correct_cat[:, -1].view(1, len(correct_outputs))[0].type('torch.LongTensor')

    loss_correctly_preds = criterion(correct_outputs, get_target_in_appropriate_format(args, targets_for_correct.cuda(),
                                                                                       num_classes))
    loss_incorrectly_preds = criterion(incorrect_outputs, get_target_in_appropriate_format(args,
                                                                                           targets_for_incorrect.cuda(),
                                                                                           num_classes))
    return loss_correctly_preds, loss_incorrectly_preds, correct_outputs, incorrect_outputs


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
        net = models.resnext50_32x4d(pretrained=False)
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


def get_network(args, num_classes):
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
        return cf.learning_rate_mtd4(args.lr, epoch)
    else:
        sys.exit("Error! Unrecognized learing rate scheduler")


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


def perform_temperature_scaling(outputs):
    if args.temp_scale_idea == 'temp_scale_default':
        Tmax = torch.max(torch.FloatTensor.abs(outputs)).item()
        #print("Tmax: ", Tmax)
        T = 1
    elif args.temp_scale_idea == 'temp_scale_idea1':
        T = torch.max(torch.FloatTensor.abs(outputs)).item()  # idea2
        Tmax = torch.max(torch.FloatTensor.abs(outputs)).item()
        #print("Tmax: ", Tmax)
    elif args.temp_scale_idea == 'temp_scale_idea2':
        T = torch.max(outputs).item()  # idea2
        Tmax = torch.max(torch.FloatTensor.abs(outputs)).item()
        #print("Tmax: ", Tmax)
    elif args.temp_scale_idea == 'temp_scale_idea3':
        Tmax = torch.max(torch.FloatTensor.abs(outputs)).item()  # idea2
        if Tmax < 4:  # idea 3
            T = 1
        else:
            T = Tmax
        Tmax = torch.max(torch.FloatTensor.abs(outputs)).item()
        #print("Tmax: ", Tmax)
    elif args.temp_scale_idea == 'temp_scale_idea4':
        T = torch.max(outputs).item() - torch.min(outputs).item()  # idea4
        Tmax = torch.max(torch.FloatTensor.abs(outputs)).item()
        #print("Tmax: ", Tmax)
    else:
        sys.exit('Error! Please select a valid temperature scaling')
    # print("\nT: ", T)
    outputs = torch.mul(outputs, 2.0 / T)
    return outputs, T, Tmax


def train_model(net, epoch, args):
    global last_saved_lr, load_from_last_best, current_lr
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    # lr = cf.learning_rate(args.lr, epoch)
    #lr = args.lr

    # metrics
    metrics = {}

    optimizer = optim.SGD(net.parameters(), lr=get_learning_rate(args, epoch), momentum=0, weight_decay=5e-4)

    # optimizer = optim.Adam(net.parameters(), lr=cf.learning_rate(lr, epoch))

    # print out current state of the learning rate
    print("\n\n param_group: ", optimizer.param_groups[0]['lr'])
    current_lr = optimizer.param_groups[0]['lr']

    # if current_lr < last_saved_lr:
    #     #load_from_last_best = True
    #     # Load last best saved model (Maybe not useful)
    #     print("\nLoading last best saved model: ", saved_model_overall_best)
    #     net = load_checkpoint(saved_model_overall_best)
    #     last_saved_lr = current_lr

    print('\n=> Training Epoch #%d' % epoch)
    for batch_idx, data in enumerate(train_loader):
        (inputs, targets), img_names = data
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # Forward Propagation
        _, predicted = torch.max(outputs.data, 1)

        #outputs_t, T, Tmax = perform_temperature_scaling(outputs)

        #print("outputs1: ", outputs)
        # esla temporarily commented out to test experimental idea below
        loss = criterion(outputs, get_target_in_appropriate_format(args, targets, num_classes))

        # esla Experimental Idea: updating loss based on NLL for the incorrectly classified for every batch
        # loss_corr, loss_incorr, _, _ = decompose_loss(outputs, targets, predicted)
        #
        # threshold = 1.2
        #
        # if (loss_incorr / loss_corr) > threshold:
        #     loss = criterion(outputs, get_target_in_appropriate_format(args, targets, num_classes))
        # else:
        #     loss = loss_incorr
        #
        # if math.isnan(loss):
        #     print("\nIgnoring nan loss")
        #     continue

        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        #if math.isnan(Tmax):
        if True:
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] \t\t Loss: %.4f Acc@1: %.3f%% '
                         % (epoch, num_epochs, batch_idx + 1,
                            (len(train_set) // batch_size) + 1, loss.item(), 100. * correct / total))
        else:
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d] |%3d |%3d\t\t Loss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx + 1,
                            (len(train_set) // batch_size) + 1, T, Tmax, loss.item(), 100. * correct / total))

        sys.stdout.flush()

    print('|\t\tLoss: %.4f Acc@1: %.3f%%' % (train_loss / batch_idx, 100. * correct / total))
    # metrics
    accuracy = 100. * correct / total
    metrics['accuracy'] = accuracy.cpu().data.numpy()
    metrics['train_loss'] = train_loss / batch_idx

    return metrics

def test_model(net, dataset_loader, epoch=None, is_validation_mode=False):
    global best_accuracy, best_acc_ace, logits, true_labels, pred_labels
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    # esla adding
    all_softmax_values = torch.FloatTensor().cuda()
    all_logits = torch.FloatTensor().cuda()
    all_targets = torch.LongTensor().cuda()
    all_preds = torch.LongTensor().cuda()
    all_img_paths = []

    # dataframe to callect all results
    columns = ['ImageNames', 'TrueLabels', 'Logits', 'SoftmaxValues', 'PredictedLabels', 'PredictedProbs']
    df = pd.DataFrame(columns=columns)

    # metrics data structure
    metrics = {}

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_loader):
            (inputs, targets), img_names = data
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            if is_validation_mode:
                #loss = criterion(outputs, get_one_hot_embedding(targets, num_classes))  # Loss for multi-label loss
                loss = criterion(outputs, get_target_in_appropriate_format(args, targets, num_classes))
                test_loss += loss.item()

            softmax_scores = F.softmax(outputs, dim=1)
            _, predicted = torch.max(softmax_scores.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # esla adding the concatenation of all logits results
            all_img_paths.extend(img_names)
            all_softmax_values = torch.cat((all_softmax_values, softmax_scores), 0)
            all_logits = torch.cat((all_logits, outputs), 0)
            all_targets = torch.cat((all_targets, targets), 0)
            all_preds = torch.cat((all_preds, predicted), 0)

            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] ' % (batch_idx, len(dataset_loader)))
            sys.stdout.flush()

        torch.cuda.synchronize()
        logits = all_logits.cpu().data.numpy().tolist()
        softmax_values = all_softmax_values.cpu().data.numpy().tolist()
        true_labels = all_targets.cpu().data.numpy().tolist()
        pred_labels = all_preds.cpu().data.numpy().tolist()

    # targets = all_targets.type('torch.FloatTensor').view(len(all_targets), 1).cuda()
    # preds = all_preds.type('torch.FloatTensor').view(len(all_preds), 1).cuda()
    #
    # temp_result = torch.cat([all_logits, targets, preds], 1)
    #
    # correct_cat = temp_result[:, :-1][temp_result[:, -1] == temp_result[:, -2]]
    # incorrect_cat = temp_result[:, :-1][temp_result[:, -1] != temp_result[:, -2]]
    #
    # incorrect_outputs = incorrect_cat[:, :-1]
    # #esla debugging
    # print("incorrect_cat[:, -1].view(1, len(incorrect_outputs))[0]",
    #       incorrect_cat[:, -1].view(1, len(incorrect_outputs))[0].shape)
    # targets_for_incorrect = incorrect_cat[:, -1].view(1, len(incorrect_outputs))[0].type('torch.LongTensor')
    #
    # correct_outputs = correct_cat[:, :-1]
    # # esla debugging
    # print("correct_cat[:, -1].view(1, len(incorrect_outputs))[0]",
    #       correct_cat[:, -1].view(1, len(correct_outputs))[0].shape)
    # targets_for_correct = correct_cat[:, -1].view(1, len(correct_outputs))[0].type('torch.LongTensor')
    #
    # loss_correctly_preds = criterion(correct_outputs, get_target_in_appropriate_format(args,targets_for_correct.cuda(),
    #                                                                                    num_classes))
    # loss_incorrectly_preds = criterion(incorrect_outputs, get_target_in_appropriate_format(args,
    #                                                                                        targets_for_incorrect.cuda(),
    #                                                                                        num_classes))

    loss_correctly_preds, loss_incorrectly_preds, _, _ = decompose_loss(all_logits, all_targets, all_preds)

    accuracy = get_topk_accuracy(all_logits, all_targets)[0].item()


    max_softmax_scores = list(np.max(softmax_values, axis=1))
    # esla debug (alternative way to get the predicted labels
    all_preds = list(all_preds.cpu().data.numpy())
    # ensure they are the same
    #assert all_preds != max_softmax_scores, 'These two must be the same'

    df['ImageNames'] = all_img_paths
    df['TrueLabels'] = true_labels
    df['SoftmaxValues'] = softmax_values
    df['Logits'] = logits
    df['PredictedLabels'] = pred_labels
    df['PredictedProbs'] = max_softmax_scores

    # compute Adaptive ECE
    ece_results = get_adaptive_ece(true_labels, pred_labels, max_softmax_scores)

    # balanced accuracy score
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    true_labels_1_hot = to_categorical(true_labels, num_classes)
    auc = roc_auc_score(true_labels_1_hot, softmax_values)

    # TBD: get this into a metrics data structure and return it if testing labeled datasets
    #accuracy = 100. * correct / total
    #accuracy = accuracy.cpu().data.numpy()
    balanced_accuracy = balanced_accuracy * 100
    auc = auc * 100

    # get the results
    metrics['accuracy'] = accuracy
    metrics['balanced_accuracy'] = balanced_accuracy
    metrics['test_loss'] = test_loss / batch_idx
    metrics['test_loss_corrects'] = loss_correctly_preds.item()
    metrics['test_loss_incorrects'] = loss_incorrectly_preds.item()
    metrics['auc'] = auc
    metrics['ece_total'] = ece_results['ece_total']
    metrics['ece_pos_gap'] = ece_results['ece_pos_gap']
    metrics['ece_neg_gap'] = ece_results['ece_neg_gap']
    
    if is_validation_mode:
        print("\n| Validation Epoch #%d\t| Loss: %.4f | Corr Loss: %.4f | Incorr Loss: %.4f | Acc@1: %.2f%% | BalAcc@1: %.2f%% "
            " | ECE_Total: %.6f | ECE_Pos: %.6f | ECE_Neg: %.6f | auc: %.2f%%" %
            (epoch, test_loss / batch_idx, loss_correctly_preds.item(), loss_incorrectly_preds.item(), accuracy, balanced_accuracy, 
             ece_results['ece_total'], ece_results['ece_pos_gap'], ece_results['ece_neg_gap'], auc))
    else:
        print("\n| \t\t\t Acc@1: %.2f%% | BalAcc@1: %.2f%% | ECE: %.6f | auc: %.2f%%" %
              (accuracy, balanced_accuracy, ece_results['ece_total'], auc))

    if is_validation_mode:
        if accuracy > best_accuracy  or accuracy  - metrics['test_loss_incorrects'] > best_acc_ace:  # stopping criteria idea 2

            if accuracy > best_accuracy:
                suffix = 'val_acc'
            else:
                suffix = 'val_acc_incorr_nll'

            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % accuracy)
            state = {
                'whole_model': net,
                'model': net.module if use_cuda else net,
                'acc': accuracy,
                'epoch': epoch,
            }

            save_point = './checkpoint/' + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + os.sep

            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            saved_model_overall_best = save_point + file_name + '.pth'
            saved_model_curr_best = save_point + file_name + '-epoch-' + str(epoch) + '.pth'
            print("Saving model: {}".format(saved_model_curr_best))
            torch.save(state, saved_model_curr_best)
            #print("Saving model: {}".format(saved_model_overall_best))
            #torch.save(state, saved_model_overall_best)
            val_csv_file = save_point + file_name + '-epoch-' + str(epoch) + '-' + suffix + '.csv'
            print("saving validation results: {}".format(val_csv_file))
            df.to_csv(val_csv_file)

            best_accuracy = accuracy  # default stopping criteria idea
            # best_accuracy = (accuracy/100) - metrics['ece_total']  # stopping criteria idea 1
            best_acc_ace = accuracy - metrics['test_loss_incorrects']  # stopping criteria idea 2

    return df, logits, true_labels, pred_labels, metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Image Classification')

    task_selection_group = parser.add_argument_group('learning task type')
    dataset_params_group = parser.add_argument_group("dataset params")
    training_group = parser.add_argument_group("training params")
    inference_only_group = parser.add_argument_group("test-only params")

    # task type arguments
    task_selection_group.add_argument('--learning_type', default='multi-class', type=str, help=""" to select the kind of
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
    training_group.add_argument('--estimate_lr', '-lre',action='store_true', help='Use LR Finder to get rough estimate of start lr')
    training_group.add_argument('--resume_from_model', '-rm', help='Model to load to resume training from')
    training_group.add_argument('--lr', default=0.001, type=float, help='learning_rate')
    training_group.add_argument('--alpha', '-a', default=None, type=float, help='alpha value for focal loss')
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
    inference_only_group.add_argument('--inference_filename', '-ifn', type=str, help='file name to save inference results')

    args = parser.parse_args()

    # Ensure the datasets root directory is valid
    if args.dataset not in ['cifar10', 'cifar100', 'ba4_project']:
        assert os.path.isdir(args.datasets_class_folders_root_dir), 'Please provide a valid root directory for all datasets'

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
    #start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
    start_epoch, num_epochs, optim_type = cf.start_epoch, cf.num_epochs, cf.optim_type

    # Optimizers
    # optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    # misc variables
    experiment_dir = datetime.now().strftime("%d-%b-%Y-%H_%M_%S.%f")

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
            #ClaheTransform(),
            #transforms.ToPILImage(),
            #transforms.Grayscale(num_output_channels=3),
            #transforms.Resize(augs.size),
            transforms.RandomResizedCrop(augs.size, scale=(0.8, 1.0)),
            torchvision.transforms.ColorJitter(hue=.05, saturation=.05),            
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(augs.mean, augs.std)
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            #ClaheTransform(),
            #transforms.ToPILImage(),
            transforms.Resize(augs.size),
            transforms.CenterCrop(augs.size),
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
    else:
        sys.exit("Error! Please provide the appropriate transform")

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

    if args.dataset == 'cifar10_orig':
        if is_training:
            print("| Preparing CIFAR-10 dataset...")
            sys.stdout.write("| ")
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=augs.no_augmentation)
            val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                   transform=augs.no_augmentation)
        num_classes = 10
        aug['size'] = 32
        assert train_set and val_set is not None, "Please ensure that you have valid train and val dataset formats"
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        #val_loader_lr_est = torch.utils.data.DataLoader(val_set_lr_est, batch_size=batch_size, shuffle=False,
        #                                                num_workers=4)
    elif args.dataset == 'cifar100_orig':
        if is_training:
            print("| Preparing CIFAR-100 dataset...")
            sys.stdout.write("| ")
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=augs.tf_transform)
            val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                    transform=augs.no_augmentation)
        num_classes = 100
        aug['size'] = 32
        assert train_set and val_set is not None, "Please ensure that you have valid train and val dataset formats"
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
        #val_loader_lr_est = torch.utils.data.DataLoader(val_set_lr_est, batch_size=batch_size, shuffle=False,
        #                                                num_workers=4)

    elif args.dataset_class_type == "class_folders":
        if is_training:
            train_root = datasets_root_dir + "/train"
            val_root = datasets_root_dir + "/val"
            if dataset_class_type == "class_folders":
                #train_set = FolderDatasetWithImgPath(train_root, transform=data_transforms['train'])
                #val_set = FolderDatasetWithImgPath(val_root, transform=data_transforms['val'])
                train_set = FolderDatasetWithImgPath(train_root, transform=train_transform)
                val_set = FolderDatasetWithImgPath(val_root, transform=val_transform)
                val_set_lr_est = torchvision.datasets.ImageFolder(train_root, transform=val_transform)
                print("class info: {}".format(train_set.class_to_idx))
            elif dataset_class_type == "csv files":
                train_csv = csv_root_dir + "/" + args.dataset + "_train.csv"
                val_csv = csv_root_dir + "/" + args.dataset + "_val.csv"
                train_set = CSVDataset(root=train_root, csv_file=train_csv, image_field='image_path', target_field='NV',
                                       transform=augs.train_transform)
                val_set = CSVDatasetWithName(root=val_root, csv_file=val_csv, image_field='image_path',
                                             target_field='NV',
                                             transform=val_transform)
            else:
                sys.exit("Should never be reached!!! Check dtaset_class_type argument")
            # Ensure all datasets for training have equal number of classes
            assert len(train_set.class_to_idx) == len(val_set.class_to_idx), 'Check train and val dataset directories'
            num_classes = len(train_set.class_to_idx)

            assert train_set and val_set is not None, "Please ensure that you have valid train and val dataset formats"
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
            val_loader_lr_est = torch.utils.data.DataLoader(val_set_lr_est, batch_size=batch_size, shuffle=False, num_workers=4)

        if is_inference:
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
            assert inference_set is not None, "Please ensure that you have valid inference dataset formats"
            inference_loader = torch.utils.data.DataLoader(inference_set, batch_size=batch_size, shuffle=False,
                                                       num_workers=4)
    else:
        sys.exit("ERROR! This place should never be reached")

    # Save experiment configurations
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    if not os.path.isdir('checkpoint/' + experiment_dir):
        #os.mkdir('checkpoint/' + experiment_dir)
        os.makedirs("checkpoint" + "/" + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + "/")

    exp_conf_file = "checkpoint" + "/" + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + "/" + "training_setting.txt"
    with open(exp_conf_file, 'a+') as f:
        f.write('\n'.join(sys.argv[1:]))
    
    # get the appropriate loss criterion for training
    #if not args.inference_only:
    
    criterion = get_loss_criterion(args, gamma=0, alpha=args.alpha)

    if args.inference_only:
        print('\n[Inference Phase] : Model setup')
        checkpoint_file = args.inference_model
        assert os.path.exists(checkpoint_file) and os.path.isfile(
            checkpoint_file), 'Error: No checkpoint directory found!'
        net_name = os.path.basename(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        net = checkpoint['model']

        if use_cuda:
            net.cuda()
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        print('\n=> Inference in Progress')
        if args.validate_train_dataset:
            all_results_df, logits, true_labels, pred_labels, metrics = test_model(net, inference_loader, is_validation_mode=True)
        else:
            all_results_df, logits, true_labels, pred_labels, metrics = test_model(net, inference_loader)

        # write results
        dataset_category = 'inference'
        if args.validate_train_dataset:
            filename = "checkpoint" + "/" + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + "/" + "inference_train_dataset.csv"
            prefix_result_file = args.dataset + "-" + str(args.input_image_size) + '_' + dataset_category + '_' + net_name + 'validated_train_dataset'
        else:
            filename = "checkpoint" + "/" + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + "/" + args.inference_filename
            prefix_result_file = args.dataset + "-" + str(args.input_image_size) + '_' + dataset_category + '_' + net_name
        with open(filename, 'a+') as infile:
            csv_writer = csv.writer(infile, dialect='excel')
            csv_writer.writerow(list(metrics.values()))

        # Save results to files
        #dataset_category = 'inference'
        #prefix_result_file = args.dataset + "-" + str(args.input_image_size) + '_' + dataset_category + '_' + net_name
        #all_results_df.to_csv(prefix_result_file + ".csv")
        all_results_df.to_csv(filename)

        if args.validate_train_dataset:
            with open(prefix_result_file + '.logits', 'wb') as f:
                pickle.dump((true_labels, pred_labels, logits), f)

        sys.exit(0)

    # To quickly visualize the effect of transforms from the dataloader
    # for i in range(500):
    #     show_dataloader_images(train_loader, augs, i, is_save=False, save_dir="./sample_images")

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


    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0
    
    if args.estimate_lr:
        assert dataset_class_type == "class_folders", 'This only works for class folder dataset type'
        # previous weight_decay = 5e-4
        # another one to try weight_decay=1e-2)
        #optimizer = optim.SGD(net.parameters(), lr=get_learning_rate(args, epoch), momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(net.parameters(), lr=1e-7)
        lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        #lr_finder.range_test(train_loader, end_lr=100, num_iter=100, step_mode="exp")
        lr_finder.range_test(train_loader, val_loader=val_loader_lr_est, end_lr=1, num_iter=50, step_mode="exp")
        lr_finder.plot()
        #print(lr_finder.history)
        filename = "checkpoint" + "/" + experiment_dir + "-" +  args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + "/" + "lr_estimate_log.csv"
        with open(filename, 'w') as outfile:
            csv_writer = csv.writer(outfile, dialect='excel')
            csv_writer.writerow(['lr', 'loss'])
            lr, loss = lr_finder.history.values()
            for i in range(len(loss)):
                #print(loss[i], lr[i])
                csv_writer.writerow([lr[i], loss[i]])

        lr_finder.reset()
        sys.exit()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()

        train_metrics = train_model(net, epoch, args)
        # validate_model(net, epoch)
        df, logits, true_labels, pred_labels, val_metrics = test_model(net, val_loader, epoch, is_validation_mode=True)

        
        # idea: update the gamma in a focal loss (Experimental)
        #print('esla debug1')
        # temporarily hard code alpha
        #alpha = 0.018
        if args.alpha:
            alpha = args.alpha + (val_metrics['balanced_accuracy'] - 50.0) / 1000  # focal loss improvement idea 1

        if args.train_loss_idea == 'ce':
            gamma = 0.0
        elif args.train_loss_idea == 'loss_idea1':
            gamma = 10*val_metrics['ece_pos_gap']   # loss idea 1
        elif args.train_loss_idea == 'loss_idea2':
            gamma = val_metrics['test_loss_incorrects']  # loss idea 2
        elif args.train_loss_idea == 'loss_idea3':
            gamma = val_metrics['test_loss_incorrects'] + 10*val_metrics['ece_total'] # loss idea 3
        elif args.train_loss_idea == 'loss_idea4':
            gamma = val_metrics['test_loss_incorrects'] + 10*val_metrics['ece_pos_gap'] # loss idea 4
        else:
            sys.exit('Error!, Choose a valid training loss idea')
        print("alpha: ", alpha)
        print("gamma:, ", gamma)

        criterion = get_loss_criterion(args, gamma=gamma, alpha=alpha)

        # write results
        filename = "checkpoint" + "/" + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + "/" + "training_log.csv"
        with open(filename, 'a+') as infile:
            csv_writer = csv.writer(infile, dialect='excel')
            if epoch == 1:
                csv_writer.writerow(['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_bal_acc', 
                                    'val_loss', 'val_corr_loss', 'val_incorr_loss', 'val_auc', 'val_ece_total', 'val_ece_pos', 'val_ece_neg'])
            csv_writer.writerow([epoch] + list(train_metrics.values()) + list(val_metrics.values()))

        filename = 'checkpoint/' + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(args.input_image_size) + '/' + args.net_type + '-' + str(epoch) + '-' + 'val'

        with open(filename + '.logits', 'wb') as f:
            pickle.dump((true_labels, pred_labels, logits), f)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))
