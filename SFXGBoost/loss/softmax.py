import numpy as np

def getLoss(y, y_pred):
    """mLogloss defualt evaluation function for softmax in the XGBoost's source code. 

    Args:
        actual (np.array): one-hot encoded loss array
        predicted (np.array): prediction probability array. 

    Returns:
        float: loss
    """
    loss = 0
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 -epsilon)

    loss  = - np.sum(y * np.log(y_pred)) / len(y)

    return loss

def getGradientHessians(y, y_pred, case_weight=None):
    if case_weight is not None: case_weight = case_weight.T
    if case_weight is None: case_weight = np.ones(y_pred.shape)
    grad = np.zeros((y_pred.shape), dtype=float) # for multi-class
    hess = np.zeros((y_pred.shape), dtype=float) # for multi-class
    for rowid in range(y_pred.shape[0]):
        wmax = max(y_pred[rowid]) # line 10s0 multiclass_obj.cu
        wsum =0.0
        for i in y_pred[rowid] : wsum +=  np.exp(i - wmax)
        for c in range(y_pred.shape[1]):
            p = np.exp(y_pred[rowid][c]- wmax) / wsum 
            target = y[rowid]
            g = p - 1.0 if c == target else p
            h = max((2.0 * p * (1.0 - p)).item(), 1e-6)
            grad[rowid][c] = g * case_weight[rowid][c]
            hess[rowid][c] = h * case_weight[rowid][c]
    return grad, hess #nUsers, nClasses

def getGradientHessiansVectorized(y, y_pred, case_weight=None):
    if case_weight is None:
        case_weight = np.ones_like(y_pred)
    
    # Step 1: Compute wmax for each row
    wmax = np.max(y_pred, axis=1, keepdims=True)
    
    # Step 2: Calculate wsum
    wsum = np.sum(np.exp(y_pred - wmax), axis=1, keepdims=True)
    
    # Step 3: Compute probabilities (p)
    p = np.exp(y_pred - wmax) / wsum
    
    # Step 4: Calculate gradients (g)
    correct_class_mask = np.eye(y_pred.shape[1])[y]
    g = p - correct_class_mask
    
    # Step 5: Calculate Hessians (h)
    # h = np.maximum(2.0 * p * (1.0 - p), 1e-6)
    h = np.maximum( p * (1.0 - p), 1e-6)
    
    # Step 6: Apply case_weight
    grad = g * case_weight
    hess = h * case_weight
    
    return grad, hess