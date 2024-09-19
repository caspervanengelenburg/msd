import torch
import numpy as np


def mIOU(pred_mask, mask, classes, smooth=1e-10):
    """
    Computes the mean Intersection-over-Union between two masks;
    the predicted multi-class segmentation mask and the ground truth.
    """

    n_classes = classes

    # make directly equipable when training (set grad off)
    with torch.no_grad():

        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for c in range(0, n_classes):  #loop over possible classes

            # compute masks per class
            true_class = pred_mask == c
            true_label = mask == c

            # when label does not exist in the ground truth, set to NaN
            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)


def classwise_iou(pred_mask, mask, num_classes, smooth=1e-10):
    """
    Computes the Intersection-over-Union for each class between two masks;
    the predicted multi-class segmentation mask and the ground truth.
    """
    with torch.no_grad():
        pred_mask = pred_mask.view(-1)
        mask = mask.view(-1)

        iou_per_class = []
        for c in range(num_classes):  # Loop over possible classes
            # Compute masks for the current class
            true_class = pred_mask == c
            true_label = mask == c

            if true_label.long().sum().item() == 0:  # No presence of the class in the ground truth
                iou_per_class.append(np.nan)
            else:
                # Calculate intersection and union for the current class
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                # Calculate IoU
                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)

        return iou_per_class

def mean_iou(pred_mask, mask, classes, smooth=1e-10):
    """
    Computes the mean Intersection-over-Union between two masks;
    the predicted multi-class segmentation mask and the ground truth.
    """

    # n_classes = len(classes)

    # make directly equipable when training (set grad off)
    with torch.no_grad():

        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for c in classes:  #loop over possible classes

            # compute masks per class
            true_class = pred_mask == c
            true_label = mask == c

            # when label does not exist in the ground truth, set to NaN
            if true_label.long().sum().item() == 0:
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)

        return np.nanmean(iou_per_class)
