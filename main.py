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

from torchvision import models

from auglib.dataset_loader import FolderDatasetWithImgPath
from auglib.augmentation import Augmentations
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName

# Metrics
from keras.utils.np_utils import to_categorical
from calibration.temp_api import get_adaptive_ece
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# esla introducing experimental loss functions
from experimental_dl_codes.from_kaggle_post_focal_loss import FocalLoss as FocalLoss1
from experimental_dl_codes.focal_loss_pytorch.focalloss import FocalLoss as FocalLoss2
from experimental_dl_codes.RetinaNet.focal_loss import FocalLoss as FocalLoss3



# Return network and file name
def get_network(args, num_classes):
    if args.net_type == 'lenet':
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif args.net_type == 'vggnet':
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-' + str(args.depth)
    elif args.net_type == 'resnet':
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-' + str(args.depth)
    elif args.net_type == 'wide-resnet':
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-' + str(args.depth) + 'x' + str(args.widen_factor)
    elif args.net_type == 'resnet18':
        file_name = 'resnet-18'
        net = models.resnet18(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        aug['size'] = 224
        aug['mean'] = [0.485, 0.456, 0.406]
        aug['std'] = [0.229, 0.224, 0.225]
        net.aug_params = aug

    elif args.net_type == 'resnet50':
        file_name = 'resnet-50'
        net = models.resnet50(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        aug['size'] = 224
        aug['mean'] = [0.485, 0.456, 0.406]
        aug['std'] = [0.229, 0.224, 0.225]
        net.aug_params = aug
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name


def train_model(net, epoch, args):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    # lr = cf.learning_rate(args.lr, epoch)
    lr = args.lr

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' % (epoch, lr))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        #print('esla1')
        #loss1 = criterion1(outputs, targets)  # Loss
        #print('esla2')
        loss2 = criterion2(outputs, targets)  # Loss
        #print('esla3')
        loss3 = criterion3(outputs, targets)  # Loss
        #print("loss ce: {}, loss f2: {}, loss f3: {}".format(loss, loss2, loss3))
        loss = loss2
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # metrics
        accuracy = 100. * correct / total
        accuracy = accuracy.cpu().data.numpy()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx + 1,
                            (len(train_set) // batch_size) + 1, loss.item(), 100. * correct / total))
        sys.stdout.flush()

    print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
          % (epoch, num_epochs, batch_idx + 1,
             (len(train_set) // batch_size) + 1, loss.item(), 100. * correct / total))

    return accuracy


def test_model(net, dataset_loader, epoch=None, is_validation_mode=False):
    global best_accuracy, logits, true_labels, pred_labels
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
                loss = criterion(outputs, targets)  # Loss
                #loss1 = criterion1(outputs, targets)  # Loss
                loss2 = criterion2(outputs, targets)  # Loss
                loss3 = criterion3(outputs, targets)  # Loss
                #print("loss ce: {}, loss f2: {}, loss f3: {}".format(loss, loss2, loss3))
                loss = loss2
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

    max_softmax_scores = list(np.max(softmax_values, axis=1))
    # esla debug (alternative way to get the predicted labels
    all_preds = list(all_preds.cpu().data.numpy())
    # ensure they are the same
    assert all_preds != max_softmax_scores, 'These two must be the same'

    df['ImageNames'] = all_img_paths
    df['TrueLabels'] = true_labels
    df['SoftmaxValues'] = softmax_values
    df['Logits'] = logits
    df['PredictedLabels'] = pred_labels
    df['PredictedProbs'] = max_softmax_scores

    # compute Adaptive ECE
    ece_results = get_adaptive_ece(true_labels, pred_labels, max_softmax_scores)
    ece = ece_results['aece']

    # balanced accuracy score
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    true_labels_1_hot = to_categorical(true_labels, num_classes=2)
    auc = roc_auc_score(true_labels_1_hot, softmax_values)

    # TBD: get this into a metrics data structure and return it if testing labeled datasets
    accuracy = 100. * correct / total
    accuracy = accuracy.cpu().data.numpy()
    balanced_accuracy = balanced_accuracy * 100
    auc = auc * 100

    # get the results
    metrics['accuracy'] = accuracy
    metrics['balanced_accuracy'] = balanced_accuracy
    metrics['auc'] = auc
    metrics['ece'] = ece

    if is_validation_mode:
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%% BalAcc@1: %.2f%% ECE: %.6f auc: %.2f%%" %
              (epoch, loss.item(), accuracy, balanced_accuracy, ece, auc))
    else:
        print("\n| \t\t\t Acc@1: %.2f%% BalAcc@1: %.2f%% ECE: %.6f auc: %.2f%%" %
              (accuracy, balanced_accuracy, ece, auc))

    if is_validation_mode:
        if accuracy > best_accuracy:

            print('| Saving Best model...\t\t\tTop1 = %.2f%%' % accuracy)
            state = {
                'model': net.module if use_cuda else net,
                'acc': accuracy,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            save_point = './checkpoint/' + args.dataset + os.sep

            if not os.path.isdir(save_point):
                os.mkdir(save_point)

            saved_model_overall_best = save_point + file_name + '.t7'
            saved_model_curr_best = save_point + file_name + '-epoch-' + str(epoch) + '.t7'
            print("Saving model: {}".format(saved_model_curr_best))
            torch.save(state, saved_model_curr_best)
            print("Saving model: {}".format(saved_model_overall_best))
            torch.save(state, saved_model_overall_best)
            val_csv_file = save_point + file_name + '-epoch-' + str(epoch) + '.csv'
            print("saving validation results: {}".format(val_csv_file))
            df.to_csv(val_csv_file)

            best_accuracy = accuracy

    return df, logits, true_labels, pred_labels, metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Image Classification')

    common_group = parser.add_argument_group("dataset params")
    training_group = parser.add_argument_group("training params")
    inference_only_group = parser.add_argument_group("test-only params")

    # common group arguments
    common_group.add_argument('--dataset_class_type', '-dct', help='The class type for the dataset')
    common_group.add_argument('--datasets_class_folders_root_dir', '-folders_dir', help='Root dir for all dataset')
    common_group.add_argument('--datasets_csv_root_dir', '-csv_dir', help='Root dir for all dataset csv files')
    common_group.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')

    training_group.add_argument('--resume_training', '-r', action='store_true', help='resume from checkpoint')
    training_group.add_argument('--resume_from_model', '-rm', help='Model to load to resume training from')
    training_group.add_argument('--lr', default=0.001, type=float, help='learning_rate')
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

    args = parser.parse_args()

    # esla extracted from code
    # input_image_size = 32
    # Ensure the datasets root directory is valid
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

    # Hyper Parameter settings
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    best_accuracy = 0
    start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

    # Transformations
    aug = dict(hflip=False, vflip=False, rotation=0, shear=0, scale=1.0, color_contrast=0, color_saturation=0,
               color_brightness=0, color_hue=0, random_crop=False, random_erasing=False, piecewise_affine=False,
               tps=False, autoaugment=False)

    aug['size'] = 224
    aug['mean'] = [0.485, 0.456, 0.406]
    aug['std'] = [0.229, 0.224, 0.225]

    augs = Augmentations(**aug)

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

    if args.dataset == 'cifar10':
        if is_training:
            print("| Preparing CIFAR-10 dataset...")
            sys.stdout.write("| ")
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=augs.no_augmentation)
            val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                   transform=augs.no_augmentation)
        num_classes = 10
    elif args.dataset == 'cifar100':
        if is_training:
            print("| Preparing CIFAR-100 dataset...")
            sys.stdout.write("| ")
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=augs.no_augmentation)
            val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=False,
                                                    transform=augs.no_augmentation)
        num_classes = 100

    elif args.dataset == "isic2019":
        if is_training:
            train_root = datasets_root_dir + "/train"
            val_root = datasets_root_dir + "/val"
            if dataset_class_type == "class folders":
                train_set = torchvision.datasets.ImageFolder(train_root, transform=augs.no_augmentation)
                val_set = FolderDatasetWithImgPath(val_root, transform=augs.no_augmentation)
                print("class info: {}".format(train_set.class_to_idx))
            elif dataset_class_type == "csv files":
                train_csv = csv_root_dir + "/" + "isic2019_train.csv"
                val_csv = csv_root_dir + "/" + "isic2019_val.csv"
                train_set = CSVDataset(root=train_root, csv_file=train_csv, image_field='image_path', target_field='NV',
                                       transform=augs.no_augmentation)
                val_set = CSVDatasetWithName(root=val_root, csv_file=val_csv, image_field='image_path',
                                             target_field='NV',
                                             transform=augs.no_augmentation)
            else:
                sys.exit("Should never be reached!!! Check dtaset_class_type argument")
            # Ensure all datasets for training have equal number of classes
            assert len(train_set.class_to_idx) == len(val_set.class_to_idx), 'Check train and val dataset directories'
            num_classes = len(train_set.class_to_idx)

            assert train_set and val_set is not None, "Please ensure that you have valid train and val dataset formats"
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

        if is_inference:
            inference_root = args.inference_dataset_dir
            if dataset_class_type == "class folders":
                inference_set = FolderDatasetWithImgPath(inference_root, transform=augs.no_augmentation)
            elif dataset_class_type == "csv files":
                inference_csv = inference_root + "/" + "isic2019_val.csv"
                inference_set = CSVDatasetWithName(root=inference_root, csv_file=inference_csv, image_field='image_path',
                                                   target_field='NV', transform=augs.no_augmentation)
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
        all_results_df, logits, true_labels, pred_labels, metrics = test_model(net, inference_loader)

        # write results
        filename = "checkpoint" +"/" + args.dataset + "/" + "inference.csv"
        with open(filename, 'a+') as infile:
            csv_writer = csv.writer(infile, dialect='excel')
            csv_writer.writerow(list(metrics.values()))

        # Save results to files
        dataset_category = 'inference'
        prefix_result_file = args.dataset + '_' + dataset_category + '_' + net_name
        all_results_df.to_csv(prefix_result_file + ".csv")

        with open(prefix_result_file + '.logits', 'wb') as f:
            pickle.dump((true_labels, pred_labels, logits), f)

        sys.exit(0)

    # Test only option
    # if args.test_only:
    #     print('\n[Test Phase] : Model setup')
    #     assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    #     _, file_name = get_network(args, num_classes)
    #     checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')
    #     net = checkpoint['model']
    #
    #     if use_cuda:
    #         net.cuda()
    #         net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #         cudnn.benchmark = True
    #
    #     print('\n=> Test in Progress')
    #     all_results_df, logits, true_labels, pred_labels = test_model(net, test_loader)
    #
    #     dataset_category = 'test'
    #     prefix_result_file = args.dataset + '_' + dataset_category + '_' + file_name
    #     with open(prefix_result_file + '.logits', 'wb') as f:
    #         pickle.dump((true_labels, pred_labels, logits), f)
    #
    #     all_results_df.to_csv(prefix_result_file + '.csv')
    #
    #     sys.exit(0)

    # Model
    print('\n[Phase 2] : Model setup')
    if args.resume_training:
        # Load checkpoint
        print('| Resuming from checkpoint...')
        # assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
        # _, file_name = get_network(args, num_classes)
        # checkpoint = torch.load('./checkpoint/' + args.dataset + os.sep + file_name + '.t7')
        # net = checkpoint['model']
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']
        _, file_name = get_network(args, num_classes)
        checkpoint_file = args.resume_from_model
        assert os.path.exists(checkpoint_file) and os.path.isfile(
            checkpoint_file), 'Error: No checkpoint directory found!'
        net_name = os.path.basename(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        net = checkpoint['model']
        start_epoch = 13
        best_accuracy = 83.88
    else:
        print('| Building net type [' + args.net_type + ']...')
        net, file_name = get_network(args, num_classes)

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    criterion1 = FocalLoss1()
    criterion2 = FocalLoss2(2)
    criterion3 = FocalLoss3(2, 1)

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + str(optim_type))

    elapsed_time = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        start_time = time.time()

        train_accuracy = train_model(net, epoch, args)
        # validate_model(net, epoch)
        df, logits, true_labels, pred_labels, metrics = test_model(net, val_loader, epoch, is_validation_mode=True)

        # write results
        filename = "checkpoint" + "/" + args.dataset + "/" + "training_log.csv"
        with open(filename, 'a+') as infile:
            csv_writer = csv.writer(infile, dialect='excel')
            if epoch == 1:
                csv_writer.writerow(['epoch', 'train acc', 'val acc', 'val bal acc', 'val auc', 'val ece'])
            csv_writer.writerow([epoch, train_accuracy] + list(metrics.values()))

        filename = 'checkpoint/' + args.dataset + '/' + args.net_type + '-' + str(epoch) + '-' + 'val'
        with open(filename + '.logits', 'wb') as f:
            pickle.dump((true_labels, pred_labels, logits), f)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))

    print('\n[Phase 4] : Testing model')
    print('* Test results : Acc@1 = %.2f%%' % (best_acc))
