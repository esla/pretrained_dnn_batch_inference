import argparse
import sys, os
import csv
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

from ast import literal_eval

import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F

from auglib.dataset_loader import FolderDatasetWithImgPath
from auglib.dataset_loader import CSVDataset, CSVDatasetWithName

from config import mean, std


def test_model(net, dataset_loader):
    global best_accuracy, best_balanced_accuracy, best_acc_ace, logits, true_labels, pred_labels
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

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_loader):
            (inputs, targets), img_names = data
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            # if is_validation_mode:
            #     #loss = criterion(outputs, get_one_hot_embedding(targets, num_classes))  # Loss for multi-label loss
            #     loss = criterion(outputs, get_target_in_appropriate_format(args, targets, num_classes))
            #     test_loss += loss.item()

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
    #assert all_preds != max_softmax_scores, 'These two must be the same'

    df['ImageNames'] = all_img_paths
    df['TrueLabels'] = true_labels
    df['SoftmaxValues'] = softmax_values
    df['Logits'] = logits
    df['PredictedLabels'] = pred_labels
    df['PredictedProbs'] = max_softmax_scores

    return df, logits, true_labels, pred_labels


def prepare_siim_submission(results_csv, submission_filenane, preprocess_image_names_func=None):
    df = pd.read_csv(results_csv)
    df_submission = pd.DataFrame()
    if preprocess_image_names_func:
        df['ImageNames'] = df['ImageNames'].apply(preprocess_image_names_func)
    df_submission['image_name'] = df['ImageNames']
    df.SoftmaxValues = df.SoftmaxValues.apply(literal_eval)
    df_softmax_values = pd.DataFrame(df['SoftmaxValues'].to_list(), columns=['pred_benign', 'pred_malignant'])
    df_submission['target'] = df_softmax_values['pred_malignant']
    df_submission.to_csv(submission_filenane, index=False)


def prepare_siim_submission_df(results_df, submission_filenane, preprocess_image_names_func=None):
    #df = pd.read_csv(results_csv)
    df = results_df
    df_submission = pd.DataFrame() 
    # To be used only when original label names were changed
    if preprocess_image_names_func:
        df['ImageNames'] = df['ImageNames'].apply(extract_approp_name)
    df_submission['image_name'] = df['ImageNames']
    df.SoftmaxValues = df.SoftmaxValues.apply(literal_eval)
    df_softmax_values = pd.DataFrame(df['SoftmaxValues'].to_list(), columns=['pred_benign', 'pred_malignant'])
    df_submission['target'] = df_softmax_values['pred_malignant']
    df_submission.to_csv(submission_filenane, index=False)


def extract_approp_name(col_elem):
    name = os.path.basename(col_elem).split('.')[0]
    return name


if __name__ == "__main__":

    # Temporary hardcoded
    batch_size = 128
    use_cuda = torch.cuda.is_available()
    experiment_dir = datetime.now().strftime("%d-%b-%Y-%H_%M_%S.%f")

    parser = argparse.ArgumentParser(description='PyTorch Image Classification')

    inference_only_group = parser.add_argument_group("test-only params")
    dataset_params_group = parser.add_argument_group("dataset params")

    # inference related arguments
    inference_only_group.add_argument('--inference_only', '-i', action='store_true',
                                      help='Make inference mode with the saved model')
    inference_only_group.add_argument('--inference_model', '-im', help='Model to load for inference')
    inference_only_group.add_argument("--inference_dataset_dir", "-idir",
                                      help="root directory for inference class folders or CSV files")
    inference_only_group.add_argument('--inference_filename', '-ifn', type=str,
                                      help='file name to save inference results')

    # dataset parameters group arguments
    dataset_params_group.add_argument('--input_image_size', type=int, help='input image size for the network')
    dataset_params_group.add_argument('--dataset_class_type', '-dct', help='The class type for the dataset')
    dataset_params_group.add_argument('--datasets_class_folders_root_dir', '-folders_dir',
                                      help='Root dir for all dataset')
    dataset_params_group.add_argument('--datasets_csv_root_dir', '-csv_dir', help='Root dir for all dataset csv files')
    dataset_params_group.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
    #dataset_params_group.add_argument('--data_transform', type=str, help='The dataset transform to be used')

    args = parser.parse_args()

    inference_root = args.inference_dataset_dir
    dataset_class_type = args.dataset_class_type
    required_image_size = args.input_image_size
    dataset_mean, dataset_std = mean[args.dataset], std[args.dataset]

    # Transforms
    inference_transform = transforms.Compose([
            transforms.CenterCrop(required_image_size),
            #transforms.Resize(required_image_size),
            transforms.ToTensor(),
            transforms.Normalize(dataset_mean, dataset_std)
        ])

    if dataset_class_type == "class_folders":
        inference_set = FolderDatasetWithImgPath(inference_root, transform=inference_transform)
    elif dataset_class_type == "csv files":
        inference_csv = inference_root + "/" + "isic2019_val.csv"
        inference_set = CSVDatasetWithName(root=inference_root, csv_file=inference_csv, image_field='image_path',
                                           target_field='NV', transform=inference_transform)
    else:
        sys.exit("Should never be reached!!! Check dtaset_class_type argument")
    # ensure the inference format matches that of training.
    # To Do: In the case of inference for unlabeled dataset, one has to present the dataset in the same
    # format as for the training and validation. This needs to be improved
    assert len(inference_set.class_to_idx) > 1, """Current implementation requires inference data to be in the same 
                                                                directory structure as the test and val datasets"""
    num_classes = len(inference_set.class_to_idx)
    assert inference_set is not None, "Please ensure that you have valid inference dataset formats"
    inference_loader = torch.utils.data.DataLoader(inference_set, batch_size=batch_size, shuffle=False, num_workers=4)

    print('\n[Inference Phase] : Model setup')
    checkpoint_file = args.inference_model
    assert os.path.exists(checkpoint_file) and os.path.isfile(
        checkpoint_file), 'Error: No checkpoint directory found!'
    selected_model_name = os.path.basename(checkpoint_file)
    result_name = os.path.dirname(checkpoint_file).split('/')[-1] + "_" + selected_model_name
    #net_name = os.path.basename(checkpoint_file)
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['model']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    print('\n=> Inference in Progress')

    all_results_df, logits, true_labels, pred_labels = test_model(net, inference_loader)

    dataset_category = 'inference'
    # filename = "inference_results" + "/" + experiment_dir + "-" + args.net_type + "-" + args.dataset + "-" + str(
    #         args.input_image_size) + "/" + args.inference_filename
    #filename = "inference_results" + "/" + net_name
    prefix_result_file = "inference_results" + "/" + args.dataset + "-" + str(args.input_image_size) + '_' + dataset_category + '_' + result_name

    # with open(filename, 'a+') as infile:
    #     csv_writer = csv.writer(infile, dialect='excel')
    #     csv_writer.writerow(list(metrics.values()))

    # Save results to files
    # dataset_category = 'inference'
    # prefix_result_file = args.dataset + "-" + str(args.input_image_size) + '_' + dataset_category + '_' + net_name
    all_results_df.to_csv(prefix_result_file + "_all_results.csv")
    #all_results_df.to_csv(filename)

    # if args.validate_train_dataset:
    #     with open(prefix_result_file + '.logits', 'wb') as f:
    #         pickle.dump((true_labels, pred_labels, logits), f)

    # For Kaggle SIIM competition
    #prepare_siim_submission(prefix_result_file + "_all_results.csv", prefix_result_file + '_submission.csv')


