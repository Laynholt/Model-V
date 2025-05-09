"""
This code is adapted from the following codes:
[1] https://github.com/Lee-Gihun/MEDIAR/blob/main/train_tools/measures.py
"""

import numpy as np
from numpy.typing import NDArray
from numba import jit
from skimage import segmentation
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any, Union

from core.logger import get_logger

__all__ = [
    "compute_batch_segmentation_f1_metrics", "compute_batch_segmentation_average_precision_metrics",
    "compute_batch_segmentation_tp_fp_fn",
    "compute_segmentation_f1_metrics", "compute_segmentation_average_precision_metrics",
    "compute_segmentation_tp_fp_fn",
    "compute_confusion_matrix", "compute_f1_score", "compute_average_precision_score"
]

logger = get_logger(__name__)
  

def compute_f1_score(
    true_positives: int,
    false_positives: int,
    false_negatives: int
) -> Tuple[float, float, float]:
    """
    Computes the precision, recall, and F1-score given the numbers of
    true positives, false positives, and false negatives.
    
    Args:
        true_positives: Number of true positive detections.
        false_positives: Number of false positive detections.
        false_negatives: Number of false negative detections.
    
    Returns:
        A tuple (precision, recall, f1_score).
    """
    if true_positives == 0:
        return 0.0, 0.0, 0.0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def compute_average_precision_score(
    true_positives: int,
    false_positives: int,
    false_negatives: int
) -> float:
    """
    Computes the average precision score using the formula:
    
        Average Precision = TP / (TP + FP + FN)
    
    If the denominator is zero, returns 0.
    
    Args:
        true_positives: Number of true positive detections.
        false_positives: Number of false positive detections.
        false_negatives: Number of false negative detections.
    
    Returns:
        Average precision score as a float.
    """
    denominator = true_positives + false_positives + false_negatives
    return 0.0 if denominator == 0 else true_positives / denominator


def compute_confusion_matrix(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[int, int, int]:
    """
    Computes the confusion matrix elements (true positives, false positives, false negatives)
    for a single image given the ground truth and predicted masks.
    
    Args:
        ground_truth_mask: Ground truth segmentation mask.
        predicted_mask: Predicted segmentation mask.
        iou_threshold: IoU threshold for matching objects.
    
    Returns:
        A tuple (TP, FP, FN).
    """
    # Determine the number of objects in the ground truth and prediction.
    num_ground_truth = np.max(ground_truth_mask)
    num_predictions = np.max(predicted_mask)

    # If no predictions were made, return zeros (with a printout for debugging).
    if num_predictions == 0:
        logger.warning("No segmentation results!")
        return 0, 0, 0

    # Compute the IoU matrix and ignore the background (first row and column).
    iou_matrix = _calculate_iou(ground_truth_mask, predicted_mask)
    # Count true positives based on optimal matching.
    true_positive_count = _calculate_true_positive(iou_matrix, iou_threshold)
    # Derive false positives and false negatives.
    false_positive_count = num_predictions - true_positive_count
    false_negative_count = num_ground_truth - true_positive_count
    return true_positive_count, false_positive_count, false_negative_count


def compute_segmentation_tp_fp_fn(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou_threshold: float = 0.5,
    return_error_masks: bool = False,
    remove_boundary_objects: bool = True
) -> Dict[str, np.ndarray]:
    """
    Computes TP, FP and FN for segmentation on a single image.
    
    If the input masks have shape (H, W), they are expanded to (1, H, W).
    For multi-channel inputs (C, H, W), each channel is processed independently, and the returned
    metrics (TP, FP, FN) are provided as NumPy arrays with shape (C,).
    
    Optionally, if return_error_masks is True, binary error masks for true positives, false positives,
    and false negatives are also returned with shape (C, H, W).
    
    Args:
        ground_truth_mask: Ground truth segmentation mask (HxW or CxHxW).
        predicted_mask: Predicted segmentation mask (HxW or CxHxW).
        iou_threshold: IoU threshold for matching objects.
        return_error_masks: Whether to also return binary error masks.
        remove_boundary_objects: Whether to remove objects that touch the image boundary.
    
    Returns:
        A dictionary with the following keys:
          - 'tp', 'fp', 'fn': arrays of shape (C,)
          - If return_error_masks is True: 'tp_mask', 'fp_mask', 'fn_mask' with shape (C, H, W)
    """
    # If the masks are 2D, add a singleton channel dimension.
    ground_truth_mask = _ensure_ndim(ground_truth_mask, 3, insert_position=0)
    predicted_mask = _ensure_ndim(predicted_mask, 3, insert_position=0)

    num_channels = ground_truth_mask.shape[0]
    true_positive_list = []
    false_positive_list = []
    false_negative_list = []
    if return_error_masks:
        true_positive_mask_list = []
        false_positive_mask_list = []
        false_negative_mask_list = []

    # Process each channel independently.
    for channel in range(num_channels):
        channel_ground_truth = ground_truth_mask[channel, ...]
        channel_prediction = predicted_mask[channel, ...]
        if np.prod(channel_ground_truth.shape) < (5000 * 5000):
            results = _process_instance_matching(
                channel_ground_truth, channel_prediction, iou_threshold,
                return_masks=return_error_masks, without_boundary_objects=remove_boundary_objects
            )
        else:
            results = _compute_patch_based_metrics(
                channel_ground_truth, channel_prediction, iou_threshold,
                return_masks=return_error_masks, without_boundary_objects=remove_boundary_objects
            )
        tp = results['tp']
        fp = results['fp']
        fn = results['fn']

        true_positive_list.append(tp)
        false_positive_list.append(fp)
        false_negative_list.append(fn)
        if return_error_masks:
            true_positive_mask_list.append(results.get('tp_mask')) # type: ignore
            false_positive_mask_list.append(results.get('fp_mask')) # type: ignore
            false_negative_mask_list.append(results.get('fn_mask')) # type: ignore

    output: Dict[str, np.ndarray] = {
        'tp': np.array(true_positive_list),
        'fp': np.array(false_positive_list),
        'fn': np.array(false_negative_list)
    }
    if return_error_masks:
        output['tp_mask'] = np.stack(true_positive_mask_list, axis=0) # type: ignore
        output['fp_mask'] = np.stack(false_positive_mask_list, axis=0) # type: ignore
        output['fn_mask'] = np.stack(false_negative_mask_list, axis=0) # type: ignore
    return output


def compute_segmentation_f1_metrics(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou_threshold: float = 0.5,
    return_error_masks: bool = False,
    remove_boundary_objects: bool = True
) -> Dict[str, np.ndarray]:
    """
    Computes F1 metrics (precision, recall, F1-score) for segmentation on a single image.
    
    If the input masks have shape (H, W), they are expanded to (1, H, W).
    For multi-channel inputs (C, H, W), each channel is processed independently, and the returned
    metrics (precision, recall, f1_score, TP, FP, FN) are provided as NumPy arrays with shape (C,).
    
    Optionally, if return_error_masks is True, binary error masks for true positives, false positives,
    and false negatives are also returned with shape (C, H, W).
    
    Args:
        ground_truth_mask: Ground truth segmentation mask (HxW or CxHxW).
        predicted_mask: Predicted segmentation mask (HxW or CxHxW).
        iou_threshold: IoU threshold for matching objects.
        return_error_masks: Whether to also return binary error masks.
        remove_boundary_objects: Whether to remove objects that touch the image boundary.
    
    Returns:
        A dictionary with the following keys:
          - 'precision', 'recall', 'f1_score': arrays of shape (C,)
          - 'tp', 'fp', 'fn': arrays of shape (C,)
          - If return_error_masks is True: 'tp_mask', 'fp_mask', 'fn_mask' with shape (C, H, W)
    """
    num_channels = ground_truth_mask.shape[0]
    precision_list = []
    recall_list = []
    f1_score_list = []
    
    results = compute_segmentation_tp_fp_fn(
        ground_truth_mask, predicted_mask,
        iou_threshold, return_error_masks,
        remove_boundary_objects
    )
    # Process each channel independently.
    for channel in range(num_channels):
        tp = results['tp'][channel]
        fp = results['fp'][channel]
        fn = results['fn'][channel]
        precision, recall, f1_score = compute_f1_score(
            tp, fp, fn # type: ignore
        )
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)

    output: Dict[str, np.ndarray] = {
        'precision': np.array(precision_list),
        'recall': np.array(recall_list),
        'f1_score': np.array(f1_score_list),
    }
    output.update(results)
    return output


def compute_segmentation_average_precision_metrics(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou_threshold: float = 0.5,
    return_error_masks: bool = False,
    remove_boundary_objects: bool = True
) -> Dict[str, np.ndarray]:
    """
    Computes the average precision (AP) for segmentation on a single image.
    
    If the input masks have shape (H, W), they are expanded to (1, H, W).
    For multi-channel inputs (C, H, W), each channel is processed independently and the returned
    metrics (average precision, TP, FP, FN) are provided as NumPy arrays with shape (C,).
    
    Optionally, if return_error_masks is True, binary error masks for true positives, false positives,
    and false negatives are also returned with shape (C, H, W).
    
    Args:
        ground_truth_mask: Ground truth segmentation mask (HxW or CxHxW).
        predicted_mask: Predicted segmentation mask (HxW or CxHxW).
        iou_threshold: IoU threshold for matching objects.
        return_error_masks: Whether to also return binary error masks.
        remove_boundary_objects: Whether to remove objects that touch the image boundary.
    
    Returns:
        A dictionary with the following keys:
          - 'avg_precision': array of shape (C,)
          - 'tp', 'fp', 'fn': arrays of shape (C,)
          - If return_error_masks is True: 'tp_mask', 'fp_mask', 'fn_mask' with shape (C, H, W)
    """
    num_channels = ground_truth_mask.shape[0]
    avg_precision_list = []
    
    results = compute_segmentation_tp_fp_fn(
        ground_truth_mask, predicted_mask,
        iou_threshold, return_error_masks,
        remove_boundary_objects
    )

    # Process each channel independently.
    for channel in range(num_channels):
        tp = results['tp'][channel]
        fp = results['fp'][channel]
        fn = results['fn'][channel]
        avg_precision = compute_average_precision_score(
            tp, fp, fn # type: ignore
        )
        avg_precision_list.append(avg_precision)
        
    output: Dict[str, np.ndarray] = {
        'avg_precision': np.array(avg_precision_list)
    }
    output.update(results)
    return output


def compute_batch_segmentation_tp_fp_fn(
    batch_ground_truth: np.ndarray,
    batch_prediction: np.ndarray,
    iou_threshold: float = 0.5,
    return_error_masks: bool = False,
    remove_boundary_objects: bool = True
) -> Dict[str, np.ndarray]:
    """
    Computes segmentation TP, FP and FN for a batch of images.
    
    Expects inputs with shape (B, C, H, W). For each image in the batch, the data is extracted
    into (C, H, W) and then processed using compute_segmentation_tp_fp_fn. The results are stacked
    so that each metric has shape (B, C). If error masks are returned, their shape will be (B, C, H, W).
    
    Args:
        batch_ground_truth: Batch of ground truth masks (BxCxHxW).
        batch_prediction: Batch of predicted masks (BxCxHxW).
        iou_threshold: IoU threshold for matching objects.
        return_error_masks: Whether to also return binary error masks.
        remove_boundary_objects: Whether to remove objects that touch the image boundary.
    
    Returns:
        A dictionary with keys:
          - 'tp', 'fp', 'fn': arrays of shape (B, C)
          - If return_error_masks is True: 'tp_mask', 'fp_mask', 'fn_mask': arrays of shape (B, C, H, W)
    """
    batch_ground_truth = _ensure_ndim(batch_ground_truth, 4, insert_position=0)
    batch_prediction = _ensure_ndim(batch_prediction, 4, insert_position=0)
    
    batch_size = batch_ground_truth.shape[0]
    tp_list = []
    fp_list = []
    fn_list = []
    if return_error_masks:
        tp_mask_list = []
        fp_mask_list = []
        fn_mask_list = []

    for i in range(batch_size):
        image_ground_truth = batch_ground_truth[i]
        image_prediction = batch_prediction[i]
        result = compute_segmentation_tp_fp_fn(
            image_ground_truth,
            image_prediction,
            iou_threshold,
            return_error_masks,
            remove_boundary_objects
        )
        tp_list.append(result['tp'])
        fp_list.append(result['fp'])
        fn_list.append(result['fn'])
        if return_error_masks:
            tp_mask_list.append(result.get('tp_mask')) # type: ignore
            fp_mask_list.append(result.get('fp_mask')) # type: ignore
            fn_mask_list.append(result.get('fn_mask')) # type: ignore

    output: Dict[str, np.ndarray] = {
        'tp': np.stack(tp_list, axis=0),
        'fp': np.stack(fp_list, axis=0),
        'fn': np.stack(fn_list, axis=0)
    }
    if return_error_masks:
        output['tp_mask'] = np.stack(tp_mask_list, axis=0) # type: ignore
        output['fp_mask'] = np.stack(fp_mask_list, axis=0) # type: ignore
        output['fn_mask'] = np.stack(fn_mask_list, axis=0) # type: ignore
    return output


def compute_batch_segmentation_f1_metrics(
    batch_ground_truth: np.ndarray,
    batch_prediction: np.ndarray,
    iou_threshold: float = 0.5,
    return_error_masks: bool = False,
    remove_boundary_objects: bool = True
) -> Dict[str, np.ndarray]:
    """
    Computes segmentation F1 metrics for a batch of images.
    
    Expects inputs with shape (B, C, H, W). For each image in the batch, the data is extracted
    into (C, H, W) and then processed using compute_segmentation_f1_metrics. The results are stacked
    so that each metric has shape (B, C). If error masks are returned, their shape will be (B, C, H, W).
    
    Args:
        batch_ground_truth: Batch of ground truth masks (BxCxHxW).
        batch_prediction: Batch of predicted masks (BxCxHxW).
        iou_threshold: IoU threshold for matching objects.
        return_error_masks: Whether to also return binary error masks.
        remove_boundary_objects: Whether to remove objects that touch the image boundary.
    
    Returns:
        A dictionary with keys:
          - 'precision', 'recall', 'f1_score', 'tp', 'fp', 'fn': arrays of shape (B, C)
          - If return_error_masks is True: 'tp_mask', 'fp_mask', 'fn_mask': arrays of shape (B, C, H, W)
    """
    batch_ground_truth = _ensure_ndim(batch_ground_truth, 4, insert_position=0)
    batch_prediction = _ensure_ndim(batch_prediction, 4, insert_position=0)
    
    batch_size = batch_ground_truth.shape[0]
    precision_list = []
    recall_list = []
    f1_score_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    if return_error_masks:
        tp_mask_list = []
        fp_mask_list = []
        fn_mask_list = []

    for i in range(batch_size):
        image_ground_truth = batch_ground_truth[i]
        image_prediction = batch_prediction[i]
        result = compute_segmentation_f1_metrics(
            image_ground_truth,
            image_prediction,
            iou_threshold,
            return_error_masks,
            remove_boundary_objects
        )
        precision_list.append(result['precision'])
        recall_list.append(result['recall'])
        f1_score_list.append(result['f1_score'])
        tp_list.append(result['tp'])
        fp_list.append(result['fp'])
        fn_list.append(result['fn'])
        if return_error_masks:
            tp_mask_list.append(result.get('tp_mask')) # type: ignore
            fp_mask_list.append(result.get('fp_mask')) # type: ignore
            fn_mask_list.append(result.get('fn_mask')) # type: ignore

    output: Dict[str, np.ndarray] = {
        'precision': np.stack(precision_list, axis=0),
        'recall': np.stack(recall_list, axis=0),
        'f1_score': np.stack(f1_score_list, axis=0),
        'tp': np.stack(tp_list, axis=0),
        'fp': np.stack(fp_list, axis=0),
        'fn': np.stack(fn_list, axis=0)
    }
    if return_error_masks:
        output['tp_mask'] = np.stack(tp_mask_list, axis=0) # type: ignore
        output['fp_mask'] = np.stack(fp_mask_list, axis=0) # type: ignore
        output['fn_mask'] = np.stack(fn_mask_list, axis=0) # type: ignore
    return output


def compute_batch_segmentation_average_precision_metrics(
    batch_ground_truth: np.ndarray,
    batch_prediction: np.ndarray,
    iou_threshold: float = 0.5,
    return_error_masks: bool = False,
    remove_boundary_objects: bool = True
) -> Dict[str, NDArray]:
    """
    Computes segmentation average precision metrics for a batch of images.
    
    Expects inputs with shape (B, C, H, W). For each image in the batch, the data is extracted
    into (C, H, W) and then processed using compute_segmentation_average_precision_metrics. The results are stacked
    so that each metric has shape (B, C). If error masks are returned, their shape will be (B, C, H, W).
    
    Args:
        batch_ground_truth: Batch of ground truth masks (BxCxHxW).
        batch_prediction: Batch of predicted masks (BxCxHxW).
        iou_threshold: IoU threshold for matching objects.
        return_error_masks: Whether to also return binary error masks.
        remove_boundary_objects: Whether to remove objects that touch the image boundary.
    
    Returns:
        A dictionary with keys:
          - 'avg_precision', 'tp', 'fp', 'fn': arrays of shape (B, C)
          - If return_error_masks is True: 'tp_mask', 'fp_mask', 'fn_mask': arrays of shape (B, C, H, W)
    """
    batch_ground_truth = _ensure_ndim(batch_ground_truth, 4, insert_position=0)
    batch_prediction = _ensure_ndim(batch_prediction, 4, insert_position=0)
    
    batch_size = batch_ground_truth.shape[0]
    avg_precision_list = []
    tp_list = []
    fp_list = []
    fn_list = []
    if return_error_masks:
        tp_mask_list = []
        fp_mask_list = []
        fn_mask_list = []

    for i in range(batch_size):
        ground_truth_mask = batch_ground_truth[i]
        prediction_mask = batch_prediction[i]
        result = compute_segmentation_average_precision_metrics(
            ground_truth_mask,
            prediction_mask, 
            iou_threshold,
            return_error_masks,
            remove_boundary_objects
        )
        avg_precision_list.append(result['avg_precision'])
        tp_list.append(result['tp'])
        fp_list.append(result['fp'])
        fn_list.append(result['fn'])
        if return_error_masks:
            tp_mask_list.append(result.get('tp_mask')) # type: ignore
            fp_mask_list.append(result.get('fp_mask')) # type: ignore
            fn_mask_list.append(result.get('fn_mask')) # type: ignore

    output: Dict[str, NDArray] = {
        'avg_precision': np.stack(avg_precision_list, axis=0),
        'tp': np.stack(tp_list, axis=0),
        'fp': np.stack(fp_list, axis=0),
        'fn': np.stack(fn_list, axis=0)
    }
    if return_error_masks:
        output['tp_mask'] = np.stack(tp_mask_list, axis=0) # type: ignore
        output['fp_mask'] = np.stack(fp_mask_list, axis=0) # type: ignore
        output['fn_mask'] = np.stack(fn_mask_list, axis=0) # type: ignore
    return output


# ===================== INTERNAL HELPER FUNCTIONS =====================

def _ensure_ndim(array: np.ndarray, target_ndim: int, insert_position: int = 0) -> np.ndarray:
    """
    Makes sure that the array has the right dimension by adding axes in front if necessary.

    Args:
        array (np.ndarray): Input array.
        target_new (int): The expected number of axes.
        insert_position (int): Where to add axes.

    Returns:
        np.ndarray: An array with the desired dimension.

    Raises:
        ValueError: If the array cannot be cast to target_ndim in a valid way.
    """
    while array.ndim < target_ndim:
        array = np.expand_dims(array, axis=insert_position)

    if array.ndim != target_ndim:
        raise ValueError(
            f"Expected ndim={target_ndim}, but got ndim={array.ndim} and shape={array.shape}"
        )

    return array


def _process_instance_matching(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou_threshold: float = 0.5,
    return_masks: bool = False,
    without_boundary_objects: bool = True
) -> Dict[str, Union[int, NDArray[np.uint8]]]:
    """
    Processes instance matching on a full image by performing the following steps:
      - Removes objects that touch the image boundary and reindexes the masks.
      - Computes the IoU matrix between instances (ignoring background).
      - Computes optimal matching via linear assignment based on the IoU matrix.
    
    If return_masks is True, binary error masks (TP, FP, FN) are also generated.
    
    Args:
        ground_truth_mask: Ground truth instance mask.
        predicted_mask: Predicted instance mask.
        iou_threshold: IoU threshold for matching.
        return_masks: Whether to generate binary error masks.
        without_boundary_objects: Whether to remove objects touching the image boundary.
    
    Returns:
        A dictionary with keys:
          - 'tp', 'fp', 'fn': integer counts.
          - If return_masks is True, also 'tp_mask', 'fp_mask', and 'fn_mask'.
    """
    # Optionally remove boundary objects.
    if without_boundary_objects:
        processed_ground_truth = _remove_boundary_objects(ground_truth_mask.astype(np.int32))
        processed_prediction = _remove_boundary_objects(predicted_mask.astype(np.int32))
    else:
        processed_ground_truth = ground_truth_mask.astype(np.int32)
        processed_prediction = predicted_mask.astype(np.int32)

    num_ground_truth = np.max(processed_ground_truth)
    num_prediction = np.max(processed_prediction)

    # If no predictions are found, return with all ground truth as false negatives.
    if num_prediction == 0:
        logger.warning("No segmentation results!")
        result = {'tp': 0, 'fp': 0, 'fn': num_ground_truth}
        if return_masks:
            tp_mask = np.zeros_like(ground_truth_mask, dtype=np.uint8)
            fp_mask = np.zeros_like(ground_truth_mask, dtype=np.uint8)
            fn_mask = np.zeros_like(ground_truth_mask, dtype=np.uint8)
            # Mark all ground truth objects as false negatives.
            fn_mask[ground_truth_mask > 0] = 1
            result.update({'tp_mask': tp_mask, 'fp_mask': fp_mask, 'fn_mask': fn_mask})
        return result

    # Compute the IoU matrix for the processed masks.
    iou_matrix = _calculate_iou(processed_ground_truth, processed_prediction)
    # Compute optimal matching pairs using linear assignment.
    matching_pairs = _compute_optimal_matching_pairs(iou_matrix, iou_threshold)

    true_positive_count = len(matching_pairs)
    false_positive_count = num_prediction - true_positive_count
    false_negative_count = num_ground_truth - true_positive_count
    result = {'tp': true_positive_count, 'fp': false_positive_count, 'fn': false_negative_count}

    if return_masks:
        # Initialize binary masks for error visualization.
        tp_mask = np.zeros_like(processed_ground_truth, dtype=np.uint8)
        fp_mask = np.zeros_like(processed_ground_truth, dtype=np.uint8)
        fn_mask = np.zeros_like(processed_ground_truth, dtype=np.uint8)

        # Record which labels were matched.
        matched_ground_truth_labels = {gt for gt, _ in matching_pairs}
        matched_prediction_labels = {pred for _, pred in matching_pairs}

        # For each matching pair, mark the intersection as true positive.
        for gt_label, pred_label in matching_pairs:
            gt_region = (processed_ground_truth == gt_label)
            prediction_region = (processed_prediction == pred_label)
            intersection = gt_region & prediction_region
            tp_mask[intersection] = 1
            # Mark parts of the ground truth not in the intersection as false negatives.
            fn_mask[gt_region & ~intersection] = 1
            # Mark parts of the prediction not in the intersection as false positives.
            fp_mask[prediction_region & ~intersection] = 1

        # Mark entire regions for objects with no match.
        all_ground_truth_labels = set(np.unique(processed_ground_truth)) - {0}
        for gt_label in (all_ground_truth_labels - matched_ground_truth_labels):
            fn_mask[processed_ground_truth == gt_label] = 1

        all_prediction_labels = set(np.unique(processed_prediction)) - {0}
        for pred_label in (all_prediction_labels - matched_prediction_labels):
            fp_mask[processed_prediction == pred_label] = 1

        result.update({'tp_mask': tp_mask, 'fp_mask': fp_mask, 'fn_mask': fn_mask})
    return result


def _compute_optimal_matching_pairs(iou_matrix: np.ndarray, iou_threshold: float) -> List[Any]:
    """
    Computes the optimal matching pairs between ground truth and predicted masks using the IoU matrix.
    
    Args:
        iou_matrix: The IoU matrix between ground truth and predicted masks.
        iou_threshold: The IoU threshold for considering a valid match.
    
    Returns:
        A list of tuples (ground_truth_label, predicted_label) representing matched pairs.
    """
    # Exclude the background (first row and column).
    iou_without_background = iou_matrix[1:, 1:]
    
    if iou_without_background.size == 0:
        return []
    
    # Determine the number of possible matches.
    num_possible_matches = min(iou_without_background.shape[0], iou_without_background.shape[1])
    
    # Create a cost matrix where lower costs indicate better matches.
    cost_matrix = -(iou_without_background >= iou_threshold).astype(np.float64) - iou_without_background / (2 * num_possible_matches)
    
    # Solve the assignment problem using the Hungarian algorithm.
    matched_ground_truth_indices, matched_prediction_indices = linear_sum_assignment(cost_matrix)
    
    # Only accept matches that meet the IoU threshold.
    matched_pairs_arr = np.stack([matched_ground_truth_indices, matched_prediction_indices], axis=1)
    
    # Filtering by IoU
    ious = iou_without_background[matched_ground_truth_indices, matched_prediction_indices]
    valid_mask = ious >= iou_threshold

    # Apply a filter and index shift (the background is missing, so +1)
    return (matched_pairs_arr[valid_mask] + 1).tolist()


def _compute_patch_based_metrics(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    iou_threshold: float = 0.5,
    return_masks: bool = False,
    without_boundary_objects: bool = True
) -> Dict[str, Union[int, NDArray[np.uint8]]]:
    """
    Computes segmentation metrics using a patch-based approach for very large images.
    
    The image is divided into fixed-size patches (e.g., 2000x2000 pixels). For each patch,
    instance matching is performed and the statistics (TP, FP, FN) are accumulated.
    If error masks are requested, they are also collected and cropped to the original image size.
    
    Args:
        ground_truth_mask: Ground truth segmentation mask.
        predicted_mask: Predicted segmentation mask.
        iou_threshold: IoU threshold for matching objects.
        return_masks: Whether to generate binary error masks.
        without_boundary_objects: Whether to remove objects that touch the image boundary.
    
    Returns:
        A dictionary with accumulated 'tp', 'fp', 'fn'. If return_masks is True,
        also includes 'tp_mask', 'fp_mask', and 'fn_mask'.
    """
    H, W = ground_truth_mask.shape
    patch_size = 2000
    
    # Calculate number of patches needed in height and width.
    num_patches_height = H // patch_size + (H % patch_size != 0)
    num_patches_width = W // patch_size + (W % patch_size != 0)
    padded_height, padded_width = patch_size * num_patches_height, patch_size * num_patches_width

    # Create padded images to ensure full patches.
    padded_ground_truth = np.zeros((padded_height, padded_width), dtype=ground_truth_mask.dtype)
    padded_prediction = np.zeros((padded_height, padded_width), dtype=ground_truth_mask.dtype)
    padded_ground_truth[:H, :W] = ground_truth_mask
    padded_prediction[:H, :W] = predicted_mask

    total_tp, total_fp, total_fn = 0, 0, 0
    if return_masks:
        padded_tp_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
        padded_fp_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
        padded_fn_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)

    # Loop over all patches.
    for i in range(num_patches_height):
        for j in range(num_patches_width):
            y_start, y_end = patch_size * i, patch_size * (i + 1)
            x_start, x_end = patch_size * j, patch_size * (j + 1)
            # Extract the patch from both ground truth and prediction.
            patch_ground_truth = padded_ground_truth[y_start:y_end, x_start:x_end]
            patch_prediction = padded_prediction[y_start:y_end, x_start:x_end]
            # Process the patch and accumulate the results.
            patch_results = _process_instance_matching(
                patch_ground_truth, patch_prediction, iou_threshold,
                return_masks=return_masks, without_boundary_objects=without_boundary_objects
            )
            total_tp += patch_results['tp']
            total_fp += patch_results['fp']
            total_fn += patch_results['fn']
            if return_masks:
                padded_tp_mask[y_start:y_end, x_start:x_end] = patch_results.get('tp_mask', 0) # type: ignore
                padded_fp_mask[y_start:y_end, x_start:x_end] = patch_results.get('fp_mask', 0) # type: ignore
                padded_fn_mask[y_start:y_end, x_start:x_end] = patch_results.get('fn_mask', 0) # type: ignore

    results: Dict[str, Union[int, np.ndarray]] = {'tp': total_tp, 'fp': total_fp, 'fn': total_fn}
    if return_masks:
        # Crop the padded masks back to the original image size.
        results.update({
            'tp_mask': padded_tp_mask[:H, :W], # type: ignore
            'fp_mask': padded_fp_mask[:H, :W], # type: ignore
            'fn_mask': padded_fn_mask[:H, :W]  # type: ignore
        })
    return results


def _remove_boundary_objects(mask: np.ndarray) -> np.ndarray:
    """
    Removes objects that touch the image boundary and reindexes the mask.
    
    A border of 2 pixels is defined around the image; any object that touches this border is removed.
    
    Args:
        mask: Segmentation mask where 0 represents the background and positive integers represent object labels.
    
    Returns:
        A reindexed mask with objects touching the boundary removed.
    """
    H, W = mask.shape
    # Create a mask with a border (value 1 in border, 0 in interior).
    border_mask = np.ones((H, W), dtype=np.uint8)
    border_mask[2:H - 2, 2:W - 2] = 0
    # Multiply the mask with the border mask to identify boundary labels.
    border_labels = np.unique(mask * border_mask)
    
    # Remove objects (set to 0) that appear in the border.
    mask[np.isin(mask, border_labels[1:])] = 0
    
    # Reindex the mask so that labels are sequential.
    new_mask, _, _ = segmentation.relabel_sequential(mask)
    return new_mask


def _calculate_true_positive(iou_matrix: np.ndarray, iou_threshold: float = 0.5) -> int:
    """
    Calculates the number of true positive instances based on the IoU matrix.
    
    Args:
        iou_matrix: IoU matrix between ground truth and predicted masks (excluding background).
        iou_threshold: IoU threshold for matching.
    
    Returns:
        The number of true positive matches.
    """
    matching_pairs = _compute_optimal_matching_pairs(iou_matrix, iou_threshold)
    return len(matching_pairs)


def _calculate_iou(ground_truth_mask: np.ndarray, predicted_mask: np.ndarray) -> NDArray[np.float32]:
    """
    Computes the Intersection over Union (IoU) matrix between ground truth and predicted masks.
    
    Args:
        ground_truth_mask: Ground truth mask with integer labels.
        predicted_mask: Predicted mask with integer labels.
    
    Returns:
        An IoU matrix of shape (num_ground_truth+1, num_prediction+1).
    """
    # Compute the overlap matrix between the two masks.
    overlap_matrix = _calculate_label_overlap(ground_truth_mask, predicted_mask)
    
    # Total number of pixels in each predicted object (sum over columns).
    pixels_per_prediction = np.sum(overlap_matrix, axis=0, keepdims=True)
    # Total number of pixels in each ground truth object (sum over rows).
    pixels_per_ground_truth = np.sum(overlap_matrix, axis=1, keepdims=True)
    
    # Compute the union for each pair.
    union_matrix = pixels_per_prediction + pixels_per_ground_truth - overlap_matrix
    
    # Avoid division by zero.
    iou = np.zeros_like(union_matrix, dtype=np.float32)
    valid = union_matrix > 0
    iou[valid] = overlap_matrix[valid] / union_matrix[valid]
    return iou


@jit(nopython=True)
def _calculate_label_overlap(mask_x: np.ndarray, mask_y: np.ndarray) -> NDArray[np.uint32]:
    """
    Computes the overlap (number of common pixels) between labels in two masks.
    
    Args:
        mask_x: First mask (integer labels with 0 as background).
        mask_y: Second mask (integer labels with 0 as background).
    
    Returns:
        An overlap matrix of shape [mask_x.max()+1, mask_y.max()+1].
    """
    flat_x = mask_x.ravel()
    flat_y = mask_y.ravel()
    
    # Create an empty overlap matrix with size based on the maximum label in each mask.
    overlap = np.zeros((1 + flat_x.max(), 1 + flat_y.max()), dtype=np.uint32)
    
    # Count overlaps for each pixel pair.
    for i in range(flat_x.shape[0]):
        overlap[flat_x[i], flat_y[i]] += 1
    return overlap
