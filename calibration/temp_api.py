import numpy as np
from .AdaptiveBinning import AdaptiveBinning


# def result_plot_ece_ding(logits_file):
#     true_labels, pred_classes, pred_probs = get_prediction_results(logits_file)
#     probability = pred_probs
#     prediction = pred_classes
#     label = true_labels
#
#     infer_results = []
#
#     for i in range(len(true_labels)):
#         correctness = (label[i] == prediction[i])
#         infer_results.append([probability[i], correctness])
#
#     # Call AdaptiveBinning.
#     AECE, AMCE, confidence, accuracy, cof_min, cof_max = AdaptiveBinning(infer_results, True)


def get_adaptive_ece(true_labels, pred_labels, max_softmax_scores):


    labels_tuples = []

    # print(max_softmax_values)
    # print(true_labels)
    # print(pred_labels)

    for i in range(len(true_labels)):
        correctness = true_labels[i] == pred_labels[i]
        labels_tuples.append([max_softmax_scores[i], correctness])

    ece_metrics, amce, confidence, accuracy, min_confidence, max_confidence = AdaptiveBinning(labels_tuples,
                                                                                       show_reliability_diagram=False)

    results = {}
    results['ece_total'] = ece_metrics['ece_total']
    results['ece_pos_gap'] = ece_metrics['ece_pos_gap']
    results['ece_neg_gap'] = ece_metrics['ece_neg_gap']
    results['amce'] = amce
    results['confidence'] = confidence
    results['accuracy'] = accuracy
    results['min_cofidence'] = min_confidence
    results['max_confidence'] = max_confidence

    return results
