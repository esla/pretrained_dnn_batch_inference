import torch
import torch.nn.functional as F


class GetTensorsForEvaluations:
    def __init__(self):
        self.all_softmax_values = torch.FloatTensor().cuda()
        self.all_logits = torch.FloatTensor().cuda()
        self.all_targets = torch.LongTensor().cuda()
        self.all_preds = torch.LongTensor().cuda()
        self.all_img_paths = []

    def update_values(self, logits_outputs, targets, img_names):
        softmax_values = F.softmax(logits_outputs, dim=1)
        _, predicted = torch.max(softmax_values.data, 1)

        self.all_img_paths.extend(img_names)
        self.all_softmax_values = torch.cat((self.all_softmax_values, softmax_values), 0)
        self.all_logits = torch.cat((self.all_logits, logits_outputs), 0)
        self.all_targets = torch.cat((self.all_targets, targets), 0)
        self.all_preds = torch.cat((self.all_preds, predicted), 0)

    def get_values_as_lists(self):
        #torch.cuda.synchronize()
        logits = self.all_logits.cpu().data.numpy().tolist()
        softmax_values = self.all_softmax_values.cpu().data.numpy().tolist()
        true_labels = self.all_targets.cpu().data.numpy().tolist()
        pred_labels = self.all_preds.cpu().data.numpy().tolist()
        return logits, softmax_values, true_labels, pred_labels

