import torch
import numpy as np


def gaussian_pick(onset, length, sigma):
    """
    Create probabilistic representation of pick in timeseries.
    Gaussian(onset, sigma).
    :param onset: The nearest sample to pick onset
    :param length: The length of the trace timeseries in samples
    :param sigma: The variance of the Gaussian distribution in samples
    :return prob_pick: 1D timeseries with probabilistic representation of pick
    """
    x = np.linspace(1, length, length)
    return np.exp(-np.power(x - onset, 2.0) / (2 * np.power(sigma, 2.0)))


# TODO: Potential enhancement would be to better integrate reduction operations with pytorch -> see torch.nn.modules.loss
def soft_assign_crossentropy(
    y_pred, y_true, from_logits=False, reduction="none", eps=1e-14
):
    """
    Compute Negative-Log-Likelihood for multiple classes (Categorical-Cross-Entropy).
    This is just the soft-assignment version of torch.nn.CrossEntropyLoss().

    Expects inputs in the form  (minibatch, C) or (minibatch, C, d_1, d_2, ..., d_K)
    with K > 1 for the K-dimensional case.

    :param y_true: Ground truth labels for all classes across n-samples (N, C, ...)
    :param y_pred: Predicted probabilities (N, C, ...)
    :param from_logits: Flag for whether tensor logits (no softmax already applied)
    :param reduction: Apply reducion function across batch dim - available ops = ('none', 'mean', 'sum')
    :param eps: Clipping output precision, range [eps, 1 - eps]. Improves float numerical stability
    :return loss: NLL of softmax(C), averaged across all samples
    """
    if len(y_pred.size()) < 2:
        raise ValueError(
            "Tensor dimensions must be >= 2. With channels as dim. 1\nSee doctring for more info."
        )
    if y_pred.size() != y_true.size():
        raise ValueError(
            f"Tensor dimensions must match, got:\ny_pred {y_pred.size()}\ny_true {y_true.size()}"
        )

    if from_logits:
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = -torch.mean(torch.sum(logsoftmax(y_pred) * y_true, 1))
    else:
        torch.clamp_(y_pred, eps, 1 - eps)
        torch.clamp_(y_true, eps, 1 - eps)
        loss = -torch.mean(torch.sum((torch.log(y_pred) * y_true), 1))

    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError("reduction operation can only be `mean`, `sum`, or `none`")
