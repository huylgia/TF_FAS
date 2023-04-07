from keras import backend as K

def macro_f1_loss(y_true, y_pred, alpha=0.0, epsilon=1e-16):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y_true (tensor): true labels of shape (batch_size, n_labels)
        y_pred (tensor): predicted probabilities of shape (batch_size, n_labels)
        epsilon (float): small constant to avoid division by zero
    
    Returns:
        cost (tensor): value of the cost function for the batch
    """
    # Convert y_true to the same type as y_pred
    y_true = K.cast(y_true, dtype=y_pred.dtype)
    
    # Compute macro soft F1 score
    tp = K.sum(y_pred * y_true, axis=0)
    fp = K.sum(y_pred * (1 - y_true), axis=0)
    fn = K.sum((1 - y_pred) * y_true, axis=0)
    tn = K.sum((1 - y_pred) * (1 - y_true), axis=0)
    
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + epsilon)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + epsilon)
    
    # Calculate costs for both classes
    cost_class1 = 1 - soft_f1_class1
    cost_class0 = 1 - soft_f1_class0
    
    # Take the average of both costs
    if alpha:
        total_cost = alpha*cost_class1 + (1-alpha)*cost_class0
    else:
        total_cost = 0.5 * (cost_class1 + cost_class0)
    cost = K.mean(total_cost)
    
    return cost