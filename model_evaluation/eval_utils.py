import sys


import torch
import torch.nn.functional as functional

import numpy as np


from sklearn.metrics import confusion_matrix


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.

    # Example

    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # Expand temperature to match the size of logits
    temperature = torch.tensor([temperature]).cuda()
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature


def get_topk_accuracy(outputs, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def get_one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    if num_classes == 1:
        return y.cuda()
    return y[labels].cuda()


def get_target_in_appropriate_format(args, targets, num_classes):
    if args.learning_type in ['multi_class', 'focal_loss_target']:
        return targets
    elif args.learning_type in ['multi_label', 'focal_loss_ohe']:
        return get_one_hot_embedding(targets, num_classes)
    else:
        sys.exit("Unknown learning task type")


def get_softmax_values(logits):
    return functional.softmax(logits, dim=1)


def get_per_class_accuracies(true_labels, pred_labels, classes_dict):
    # Accuracy per class
    cm = confusion_matrix(true_labels, pred_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_per_class_vals = cm.diagonal()
    acc_per_class = {}
    for i, class_name in enumerate(classes_dict):
        acc_per_class[class_name] = 100.0 * round(acc_per_class_vals[i], 2)

    return acc_per_class


def decompose_loss(logits, targets, predictions, criterion, num_classes, args):
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
