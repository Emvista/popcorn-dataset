import torch


class NERLoss(torch.nn.Module):
    def __init__(self):
        super(NERLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, groundtruth, mask):
        """_summary_

        Args:
            predictions (torch.tensor): Predictions of the model in the shape of Batch x Max Token x Max Token x Prediction for each class
            groundtruth (_type_): Wanted class for each span (Batch x Max Token x Max Token)
            mask (_type_): Possible spans, 1 if possible else 0 (Batch x Max Token x Max Token)

        Returns:
            _type_: _description_
        """
        loss = self.loss(predictions, groundtruth).sum(-1)
        masked_loss = loss * mask
        summed_loss = masked_loss.sum()

        return summed_loss

    def evaluate(self, output, groundtruth):
        TP = torch.sum((groundtruth > 0.5) & (output > 0))
        FP = torch.sum((groundtruth < 0.5) & (output > 0))
        FN = torch.sum((groundtruth > 0.5) & (output <= 0))
        return TP, FP, FN


class NLPLoss(NERLoss):
    def __init__(self):
        super(NERLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, nlp_labels, mask):
        """_summary_

        Args:
            predictions (torch.tensor): Predictions of the model in the shape of Batch x Max Token x Max Token x Prediction for each class
            nlp_labels (torch.tensor): Wanted class for each span (Batch x Max Token x Max Token x  Interactions)
            mask (torch.tensor): Possible spans, 1 if possible else 0 (Batch x Max Token x Max Token)

        Returns:
            _type_: _description_
        """
        loss = self.loss(predictions, nlp_labels)
        masked_loss = loss * mask
        summed_loss = masked_loss.sum()
        return summed_loss


class PRUNERLoss(NERLoss):
    def __init__(self):
        super(NERLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, predictions, span_labels, mask):
        """_summary_

        Args:
            predictions (torch.tensor): Predictions of the model in the shape of Batch x Max Token x Max Token x Prediction for each class
            groundtruth (torch.tensor): Wanted class for each span (Batch x Max Token x Max Token x Interactions)
            mask (torch.tensor): Possible spans, 1 if possible else 0 (Batch x Max Token x Max Token)

        Returns:
            _type_: _description_
        """
        loss = self.loss(predictions.squeeze(), span_labels)
        masked_loss = loss * mask
        summed_loss = masked_loss.sum()
        return summed_loss

    def evaluate(self, output, groundtruth):
        TP = torch.sum((groundtruth > 0.5) & (output > 0))
        FP = torch.sum((groundtruth < 0.5) & (output > 0))
        FN = torch.sum((groundtruth > 0.5) & (output <= 0))
        return TP, FP, FN
