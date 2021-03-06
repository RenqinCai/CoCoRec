import torch

def get_recall(indices, targets, mask):
    """
    Calculates the recall score for the given predictions and targets

    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
        mask : warm start

    Returns:
        recall (float): the recall score
    """

    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices)
        
    hits *= mask.view(-1, 1).expand_as(indices)
    hits = hits.nonzero()
    
    recall = float(hits.size(0)) / float( mask.int().sum() )

    return recall


def get_mrr(indices, targets, mask):
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
        mask : warm start

    Returns:
        mrr (float): the mrr score
    """

    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices)
    hits *= mask.view(-1, 1).expand_as(indices)
    
    hits = hits.nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / float( mask.int().sum() )
    return mrr.item()


def evaluate(indices, targets, mask, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.
        mask: warm start

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """

    _, indices = torch.topk(indices, k, -1)

    indices = indices.cpu()
    targets = targets.cpu()
   
    recall = get_recall(indices, targets, mask)
    mrr = get_mrr(indices, targets, mask)
    return recall, mrr
