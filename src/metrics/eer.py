import torch


def compute_eer(scores, labels):
    thresholds = torch.sort(scores).values

    best = 1.0

    for t in thresholds:
        predict = (scores >= t).long()

        fp = ((predict == 1) & (labels == 0)).sum().float()
        fn = ((predict == 0) & (labels == 1)).sum().float()

        far = fp / (labels == 0).sum()
        frr = fn / (labels == 1).sum()

        if torch.abs(far - frr) < best:
            best = torch.abs(far - frr)
            eer = (far + frr) / 2

    return eer.item()