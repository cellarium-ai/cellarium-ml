import numpy as np
from scipy.stats import linregress

def batch_linear_regression(x_bn, y_bn):
    """
    Perform batch-mode one-dimensional linear regression on data with arbitrary batch dimensions.
    
    For each batch (with indices b), we assume a linear model:
    
        y_bn = slope_b * x_bn + intercept_b + error_bn
    
    where:
      - x_bn and y_bn are arrays of shape (..., n), with the rightmost dimension (n) 
        representing the samples,
      - the leftmost dimensions (possibly several) are treated as batch dimensions.
    
    Parameters
    ----------
    x_bn : np.ndarray
        Array of covariates with shape (..., n).
    y_bn : np.ndarray
        Array of responses with shape (..., n).
        
    Returns
    -------
    slope_b : np.ndarray
        Array of slopes with shape equal to the batch shape.
        For any batch where the variance of x_bn is zero, the slope is set to 0.
    intercept_b : np.ndarray
        Array of intercepts with shape equal to the batch shape.
    r_squared_b : np.ndarray
        Array of R-squared values with shape equal to the batch shape.
        For any batch where the total variance of y_bn is zero, r_squared_b is defined as 1.
    """
    # Assert that inputs are numpy arrays of the same shape.
    assert isinstance(x_bn, np.ndarray), "x_bn must be a numpy array"
    assert isinstance(y_bn, np.ndarray), "y_bn must be a numpy array"
    assert x_bn.shape == y_bn.shape, "x_bn and y_bn must have the same shape"
    assert x_bn.ndim >= 1, "x_bn and y_bn must have at least one dimension (samples)"

    # Compute the means along the sample dimension (rightmost axis) and keep that dimension.
    # The resulting arrays have shape (..., 1) and are named with a '_b1' suffix.
    x_mean_b1 = np.mean(x_bn, axis=-1, keepdims=True)  # shape: (..., 1)
    y_mean_b1 = np.mean(y_bn, axis=-1, keepdims=True)  # shape: (..., 1)
    
    # Compute the covariance for each batch:
    #   cov_b = sum_n[(x_bn - x_mean_b1) * (y_bn - y_mean_b1)]
    cov_b = np.sum((x_bn - x_mean_b1) * (y_bn - y_mean_b1), axis=-1)  # shape: batch shape
    
    # Compute the variance of x for each batch:
    #   var_x_b = sum_n[(x_bn - x_mean_b1)^2]
    var_x_b = np.sum((x_bn - x_mean_b1)**2, axis=-1)  # shape: batch shape
    
    # Compute the slope for each batch, using np.divide to avoid division by zero.
    slope_b = np.divide(cov_b, var_x_b, out=np.zeros_like(cov_b), where=(var_x_b != 0))
    
    # Compute the intercept for each batch.
    intercept_b = y_mean_b1.squeeze(-1) - slope_b * x_mean_b1.squeeze(-1)
    
    # Compute the predicted responses for each batch:
    #   yhat_bn = slope_b * x_bn + intercept_b (broadcasting over the sample dimension)
    yhat_bn = slope_b[..., None] * x_bn + intercept_b[..., None]
    
    # Compute the residual sum of squares (SS_res) and total sum of squares (SS_tot) for each batch.
    ss_res_b = np.sum((y_bn - yhat_bn)**2, axis=-1)
    ss_tot_b = np.sum((y_bn - y_mean_b1)**2, axis=-1)
    
    # Compute R-squared for each batch. If ss_tot_b == 0, define R-squared as 1.
    r_squared_b = np.where(ss_tot_b != 0, 1 - ss_res_b / ss_tot_b, 1.0)
    
    return slope_b, intercept_b, r_squared_b

def test_batch_linear_regression():
    """
    Test the batch_linear_regression function by comparing its outputs with:
      1. Running each batch individually using scipy.stats.linregress.
      2. Ensuring that processing the batches together or one at a time yield the same results.
    """
    # Set a random seed for reproducibility.
    np.random.seed(0)
    
    # Define a batch shape (can be multidimensional) and number of samples.
    batch_shape = (3, 4)  # Two batch dimensions for demonstration.
    num_samples = 100     # Rightmost dimension: sample index.
    
    # Generate random covariate data with shape (3, 4, 100).
    x_bn = np.random.rand(*batch_shape, num_samples)
    
    # Define true slopes and intercepts for each batch.
    true_slopes = np.linspace(0.5, 2.0, num=np.prod(batch_shape)).reshape(*batch_shape, 1)
    true_intercepts = np.linspace(1.0, 3.0, num=np.prod(batch_shape)).reshape(*batch_shape, 1)
    
    # Generate responses with added noise.
    noise_bn = np.random.randn(*batch_shape, num_samples) * 0.1
    y_bn = true_intercepts + true_slopes * x_bn + noise_bn
    
    # Use our batch regression implementation.
    slope_b, intercept_b, r_squared_b = batch_linear_regression(x_bn, y_bn)
    
    # Now, compute the regression parameters one batch at a time using scipy.stats.linregress.
    # We'll flatten the batch dimensions for easier iteration.
    x_bn_flat = x_bn.reshape(-1, num_samples)
    y_bn_flat = y_bn.reshape(-1, num_samples)
    
    slopes_scipy = np.empty(x_bn_flat.shape[0])
    intercepts_scipy = np.empty(x_bn_flat.shape[0])
    r_squared_scipy = np.empty(x_bn_flat.shape[0])
    
    for i in range(x_bn_flat.shape[0]):
        x_n = x_bn_flat[i]
        y_n = y_bn_flat[i]
        result = linregress(x_n, y_n)
        slopes_scipy[i] = result.slope
        intercepts_scipy[i] = result.intercept
        r_squared_scipy[i] = result.rvalue**2  # R-squared is the square of the r-value.
    
    # Reshape the individual results to the original batch shape.
    slopes_scipy = slopes_scipy.reshape(batch_shape)
    intercepts_scipy = intercepts_scipy.reshape(batch_shape)
    r_squared_scipy = r_squared_scipy.reshape(batch_shape)
    
    # Assert that the batch regression results match the individual regressions.
    assert np.allclose(slope_b, slopes_scipy, atol=1e-6), "Slopes do not match between batch and individual regressions."
    assert np.allclose(intercept_b, intercepts_scipy, atol=1e-6), "Intercepts do not match between batch and individual regressions."
    assert np.allclose(r_squared_b, r_squared_scipy, atol=1e-6), "R-squared values do not match between batch and individual regressions."
    
    print("Test passed: Batch regression results match individual regressions and scipy.linregress results.")
