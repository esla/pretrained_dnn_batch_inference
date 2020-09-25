from keras.utils.np_utils import to_categorical
from calibration.temp_api import get_adaptive_ece
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import pandas as pd

from model_evaluation.eval_utils import *


class ModelEvaluator:
    def __init__(self, criterion, classes_dict, other_args, with_image_names=True):
        # Note: this setup assumes all training was done on Cuda device
        # To Do: Assert that training is done on Cuda first
        self.other_args = other_args
        self.criterion = criterion
        self.include_image_names = with_image_names
        self.all_softmax_values = torch.FloatTensor().cuda()
        self.all_logits = torch.FloatTensor().cuda()
        self.all_targets = torch.LongTensor().cuda()
        self.all_preds_labels = torch.LongTensor().cuda()
        self.all_preds_probs = torch.LongTensor().cuda()
        self.all_img_names = []
        self.classes_dict = classes_dict

    def update_results(self, logits, true_labels, image_names):
        # compute secondary values
        softmax_values = get_softmax_values(logits)
        pred_softmax_probs, pred_softmax_labels = torch.max(softmax_values, 1)

        # Update the results
        if self.include_image_names:
            self.all_img_names.extend(image_names)
        self.all_softmax_values = torch.cat((self.all_softmax_values, softmax_values), 0)
        self.all_logits = torch.cat((self.all_logits, logits), 0)
        self.all_targets = torch.cat((self.all_targets, true_labels), 0)
        self.all_preds_probs = torch.cat((self.all_preds_probs, pred_softmax_probs), 0)
        self.all_preds_labels = torch.cat((self.all_preds_labels, pred_softmax_labels), 0)

    @property
    def compute_all_metrics(self):
        torch.cuda.synchronize()
        logits = self.all_logits.cpu().data.numpy().tolist()
        softmax_values = self.all_softmax_values.cpu().data.numpy().tolist()
        true_labels = self.all_targets.cpu().data.numpy().tolist()
        pred_labels = self.all_preds_labels.cpu().data.numpy().tolist()
        pred_probs = self.all_preds_probs.cpu().data.numpy().tolist()

        # Accuracies
        # Top-1 accuracies per class
        acc_per_class = get_per_class_accuracies(true_labels, pred_labels, self.classes_dict)
        # Top-k accuracies
        accuracy_top1 = get_topk_accuracy(self.all_logits, self.all_targets)[0].item()

        # Balanced accuracy
        balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
        balanced_accuracy = balanced_accuracy * 100

        # ROC Area Under Curve
        true_labels_1_hot = to_categorical(true_labels, len(self.classes_dict))
        auc = roc_auc_score(true_labels_1_hot, softmax_values)
        auc = auc * 100

        # Losses
        total_loss = self.criterion(self.all_logits, self.all_targets)
        loss_correctly_preds, loss_incorrectly_preds, _, _ = decompose_loss(self.all_logits,
                                                                            self.all_targets,
                                                                            self.all_preds_labels,
                                                                            self.criterion,
                                                                            self.classes_dict,
                                                                            self.other_args)

        # compute Adaptive ECE
        ece_results = get_adaptive_ece(true_labels, pred_labels, pred_probs)

        # Gather results
        raw_values_df = pd.DataFrame()

        if self.include_image_names:
            raw_values_df['ImageNames'] = self.all_img_names
        raw_values_df['TrueLabels'] = true_labels
        raw_values_df['SoftmaxValues'] = softmax_values
        raw_values_df['Logits'] = logits
        raw_values_df['PredictedLabels'] = pred_labels
        raw_values_df['PredictedProbs'] = pred_probs

        evaluation_metrics = dict(
            accuracy=accuracy_top1,
            balanced_accuracy=balanced_accuracy,
            test_loss=total_loss,
            test_loss_corrects=loss_correctly_preds.item(),
            test_loss_incorrects=loss_incorrectly_preds.item(),
            auc=auc,
            ece_total=ece_results['ece_total'],
            ece_pos_gap=ece_results['ece_pos_gap'],
            ece_neg_gap=ece_results['ece_neg_gap']
        )

        return raw_values_df, evaluation_metrics, acc_per_class
