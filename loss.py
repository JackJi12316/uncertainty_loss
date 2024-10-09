from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange


def clamped(x: torch.Tensor, clamp: float = 10.0) -> torch.Tensor:
    r"""This function clamps the input tensor to be between -clamp and clamp and
    then exponentiates it.

    Args:
        x (Tensor): The input tensor.
        clamp (float): The clamp value.

    Returns:
        (Tensor) The clamped and exponentiated tensor.
    """
    return torch.exp(torch.clamp(x, -clamp, clamp))


def maxnorm_loss(
        x: torch.Tensor,
        target: torch.Tensor,
        reg_factor: float = 0.0,
        p_norm: int = 4,
        reduction: Optional[str] = "mean",
        ignore_index: int = -100
) -> torch.Tensor:
    r"""Computes the max norm uncertainty loss.

    Args:
        x (Tensor): model output.
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc. The shape is
            (N,C, d1,...,dk) where N is the number of samples, C is the
            number of classes and d1,...,dk optional additional
            dimensions.
        target (Tensor): Ground truth class indicies or class probabilities.
        reg_factor (float): The weight of regularization factor. If 0, no regularization is applied.
        p_norm (int): The p-norm to use for the max norm approximation.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The max norm uncertainty loss over the batch.
    """
    evidence = clamped(x)
    target, mask = _enforce_same_dim(target=target, evidence=evidence, ignore_index=ignore_index)
    evidence = rearrange(evidence, 'b c d ->(b c) d')
    target = rearrange(target, 'b c d -> (b c) d')
    loss = dirichlet_pnorm_loss(evidence, target, p_norm=p_norm, reduction=reduction, ignore_index=ignore_index,
                                mask_ignore=mask)
    if reg_factor > 0:
        fisher_loss = dirichlet_fisher_regulizer(evidence, target, reduction=reduction, ignore_index=ignore_index,
                                                 mask_ignore=mask)
        loss += reg_factor * fisher_loss
    return loss


def dirichlet_pnorm_loss(
        evidence: torch.Tensor, target: torch.Tensor, mask_ignore, p_norm=4, reduction: Optional[str] = "mean",
        ignore_index: int = -100
) -> torch.Tensor:
    r"""This criterion computes the mean dirichlet max norm loss between the input
    and target.

    The implementation is performed in log space so we use torch's `lgamma` and
    to avoid numerical issues.  We first compute the log of the loss function and
    then exponentiate it to get the loss.  As usual prods are replaced with sums
    and divisons are replaced with subtractions and sums are replaced with logsumexp.
    It may be helpful to understand this implementation by writing down the mathematics
    of log(F) where F is the loss function from the paper.

    Reference:
        * Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation # noqa
        * https://arxiv.org/pdf/1910.04819.pdf

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc. The shape is
            (N,C, d1,...,dk) where N is the number of samples, C is the
            number of classes and d1,...,dk optional additional
            dimensions.
        target (Tensor): Ground truth class indicies or class probabilities.
        p_norm (int): The p-norm to use for the max norm approximation.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The mean dirichlet max norm loss over the batch
    """
    p = p_norm
    lgamma = torch.lgamma

    alpha = evidence + 1

    # mask out correct class in alpha
    mask = target > 0

    alpha_hat = alpha.clone()
    alpha_hat[mask] = 0

    # prepare to sum over all indicies except the correct class
    #
    # lgamma(1) = 0, so the below ensures that when we take sum(alpha + p) we get 0 in the correct class entry is what we want since we really should be summing over all classes except the correct class.
    alpha_negp = alpha.clone()
    alpha_negp[mask] = -p + 1

    s_alpha = torch.sum(alpha, dim=1, keepdim=True)
    s_alpha_hat = torch.sum(alpha_hat, dim=1, keepdim=True)

    factored_term = torch.squeeze(
        lgamma(s_alpha) - lgamma(s_alpha + p)
    )  # (batch_size,d1, d2, ..., dn) # noqa
    logsumexp_term_0 = lgamma(s_alpha_hat + p) - lgamma(
        s_alpha_hat
    )  # (batch_size, 1, d1, d2, ..., dn) # noqa
    logsumexp_term_1 = lgamma(alpha_negp + p) - lgamma(alpha_hat)

    lse = torch.cat(
        [logsumexp_term_0, logsumexp_term_1], dim=1
    )  # (batch_size, 2, d1, d2, ..., dn) # noqa
    logsumexp_term = torch.logsumexp(lse, dim=1)  # (batch_size, d1, d2, ..., dn)
    loss = torch.exp((factored_term + logsumexp_term) / p)
    reducer = _get_reducer(reduction)
    if ignore_index >= 0:
        mask_all = target[:, ignore_index:ignore_index + 1].to(torch.bool)
        mask_all = rearrange(mask_all, 'b c ->(b c)')
        if reduction == "none":
            loss = torch.where(~mask_all, loss, torch.tensor(0.0, device=loss.device))
        else:
            loss = loss[~mask_all]
    else:
        if reduction == "none":
            loss = torch.where(~mask_ignore, loss, torch.tensor(0.0, device=loss.device))
        else:
            loss = loss[~mask_ignore]
    return reducer(loss)


def dirichlet_fisher_regulizer(
        evidence: torch.Tensor, target: torch.Tensor, mask_ignore, reduction: Optional[str] = "mean",
        ignore_index: int = -100
) -> torch.Tensor:
    r"""This criterion computes the mean dirichlet fisher regulizer between the input
    and target.

    Reference:
        * Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation # noqa
        * https://arxiv.org/pdf/1910.04819.pdf

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc. The shape is
            (N,C, d1,...,dk) where N is the number of samples, C is the
            number of classes and d1,...,dk optional additional
            dimensions.
        target (Tensor): Ground truth class indicies or class probabilities.
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none', None]. If None or 'none' no reduction
            is performed. Default is "mean".

    Returns:
        (Tensor) The mean dirichlet fisher regulizer over the batch
    """
    alpha = evidence + 1

    alpha_hat = alpha.clone()
    mask = target > 0

    # prepare to sum over all indicies except the correct class
    #
    # we need sum(alpha-1) over all classes except the correct class
    # so we do the subtraction and then set the correct class entry to 0
    alpha_minus_one = alpha.clone()
    alpha_minus_one = alpha_minus_one - 1
    alpha_minus_one[mask] = 0

    s_alpha_hat = torch.sum(alpha_hat, dim=1, keepdim=True)
    polygamma_term = torch.polygamma(1, alpha_hat) - torch.polygamma(1, s_alpha_hat)
    prod = torch.square(alpha_minus_one) * polygamma_term
    reducer = _get_reducer(reduction)
    reg = 0.5 * torch.sum(prod, dim=1)
    if ignore_index >= 0:
        mask_all = target[:, ignore_index:ignore_index + 1].to(torch.bool)
        mask_all = rearrange(mask_all, 'b c ->(b c)')
        if reduction == "none":
            reg = torch.where(~mask_all, reg, torch.tensor(0.0, device=reg.device))
        else:
            reg = reg[~mask_all]
    else:
        if reduction == "none":
            reg = torch.where(~mask_ignore, reg, torch.tensor(0.0, device=reg.device))
        else:
            reg = reg[~mask_ignore]
    return reducer(reg)


def _get_reducer(reduction: Optional[str] = "mean"):
    """Returns a reducer function for the given reduction type.

    Args:
        reduction (str): The reduction type.  Must be one of
            ['mean', 'sum', 'none']. If 'none' no reduction
            is performed. Default is 'mean'.
    Returns:
        (Callable) A reducer function.
    """
    if reduction == "mean":
        return torch.mean
    elif reduction == "sum":
        return torch.sum
    elif reduction == "none":
        return lambda x: x
    else:
        raise ValueError(
            f"reduction must be one of 'mean', 'sum', or 'none' got {reduction}"
        )


def _enforce_same_dim(target: torch.Tensor, evidence: torch.Tensor, ignore_index):
    mask = None

    if target.dim() != evidence.dim():
        tgt = target.clone()
        tgt = rearrange(tgt, "b c -> (b c)")

        if ignore_index >= 0:
            tgt = F.one_hot(tgt.reshape(-1), num_classes=evidence.shape[2])
        else:
            mask = (tgt == ignore_index)
            tgt = torch.where(~mask, tgt, torch.tensor(0, device=tgt.device))
            tgt = F.one_hot(tgt.reshape(-1), num_classes=evidence.shape[2])
            tgt[mask] = 0

        tgt = rearrange(tgt, "(b m) v -> b m v", b=evidence.shape[0])
        return tgt, mask
    else:
        return target, None


def entropy(y_proba: torch.Tensor, normalize=False):
    _entropy = -torch.sum(y_proba * torch.log(y_proba), dim=1)
    if normalize:
        max_entropy = torch.log(torch.tensor(y_proba.shape[1]))
        _entropy = _entropy / max_entropy
    return _entropy


def uncertainty(x: torch.Tensor, normalize=False):
    """Computes the predictive entropy from the class probabilities.

    Args:
        x (Tensor): The class probabilities of shape (N,C, d1,...,dk).
            where N is the number of samples, C is the number of classes and
            d1,...,dk optional additional dimensions of the output.
    Returns:
        (Tensor) The uncertainty scores of shape (N, d1,...,dk).
    """
    evidence = clamped(x)
    alpha = evidence + 1
    y_proba = alpha / torch.sum(alpha, dim=1, keepdim=True)
    return entropy(y_proba, normalize=normalize)


def model_uncertainty(x: torch.Tensor):
    """Computes the epistemic (model or knowledge) uncertainty.

    Args:
        x (Tensor): model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc. The shape is
            (N,C, d1,...,dk) where N is the number of samples, C is the
            number of classes and d1,...,dk optional additional
            dimensions.


    Returns:
        (Tensor) The model uncertainty scores of shape (N, d1,...,dk).

    """
    evidence = clamped(x)
    total_uncertainty = uncertainty(evidence)
    alpha = evidence + 1
    s_alpha = torch.sum(alpha, dim=1, keepdim=True)
    epsilon = 1e-8
    div_term = alpha / (s_alpha + epsilon)
    gamma_term = torch.digamma(s_alpha + 1) - torch.digamma(alpha + 1)
    return total_uncertainty - torch.sum(div_term * gamma_term, dim=1)


def data_uncertainty(evidence: torch.Tensor):
    r"""Computes aleatoric (data) uncertainty from the model evidence.

    Args:
        evidence (Tensor): Evidence of model output.  The evidence is any non-negative
            transformation math:`g(x)` of the raw unnormalized model outputs
            math:`x`.  For example `g=relu(x)`, `g=exp(x)` etc. The shape is
            (N,C, d1,...,dk) where N is the number of samples, C is the
            number of classes and d1,...,dk optional additional
            dimensions.

    Returns:
        A tensor of shape (batch_size, d1,...dk), uncertainty score for
        each element in the batch.
    """
    return uncertainty(evidence) - model_uncertainty(evidence)


def relu_evidence(y):
    return F.relu(y)


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(target, alpha, num_classes, reg_factor, device):
    target = target.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(target, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        reg_factor,
    ).to(device)

    kl_alpha = (alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_mse_loss(
        output_hat: torch.Tensor,
        output: torch.Tensor,
        device: torch.device,
        reg_factor: float = 0.0,
        ignore_index: int = -100,
        reduction: str = "mean"
) -> torch.Tensor:
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    evidence = relu_evidence(flat_hat)
    alpha = evidence + 1
    flat = rearrange(output, "b l -> (b l)")

    target, mask_ignore = _enforce_same_dim(evidence=evidence, target=flat, ignore_index=ignore_index)
    reducer = _get_reducer(reduction)
    loss = mse_loss(target, alpha, flat_hat.size(1), reg_factor, device=device)
    if ignore_index >= 0:
        mask_all = target[:, ignore_index:ignore_index + 1].to(torch.bool)
        mask_all = rearrange(mask_all, 'b c ->(b c)')
        if reduction == "none":
            loss = torch.where(~mask_all, loss, torch.tensor(0.0, device=loss.device))
        else:
            loss = loss[~mask_all]
    else:
        if reduction == "none":
            loss = torch.where(~mask_ignore, loss, torch.tensor(0.0, device=loss.device))
        else:
            loss = loss[~mask_ignore]
    return reducer(loss)


def compute_u(x: torch.Tensor, classification_size: int, reduction: str = "mean") -> torch.Tensor:
    max_out = torch.max(x, dim=1, keepdim=True).values
    min_out = torch.min(x, dim=1, keepdim=True).values
    out_section = max_out - min_out
    symbol_alpha = clamped(x) + 1
    symbol_S = symbol_alpha.sum(dim=1)
    epsilon = 1e-8  # 防止除零
    symbol_u = classification_size / (symbol_S + epsilon)
    out_section = rearrange(out_section, "b l -> (b l)")
    if reduction == "mean":
        return symbol_u * out_section
    else:
        return symbol_u


def ce_loss(
        output_hat: torch.Tensor,
        output: torch.Tensor,
        ignore_idx: int = -100,
        reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int): index of ignore classified
        reduction (str): the way to reduce loss

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss
