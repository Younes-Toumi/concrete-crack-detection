import numpy as np
import cv2


def convert_grayscale(X: np.ndarray) -> np.ndarray:
    """
    Convert the RGB images to grayscale according to poynton weights.
    
    Parameters
    ----------
    X : np.ndarray
        The input images (N, 227, 227, 3).
    
    Returns
    -------
    np.ndarray
        The grayscale images.
    """
    poynton_weights = np.array([[0.2125, 0.7154, 0.0721]], dtype=np.float16).T


    # Stack grayscale image to create 3-channel grayscale (N, 227, 227, 3)
    grayscale = np.dot(X, poynton_weights).astype(np.uint8)  # (N, 227, 227, 1)
    
    return grayscale


def convert_sobel_edges(X_grayscale: np.ndarray) -> np.ndarray:
    """
    Detect edges in the grayscale images according to the Sobol method.
    
    Parameters
    ----------
    X_grayscale : np.ndarray
        The input grayscale images (N, 227, 227, 1).
    
    Returns
    -------
    edge_images : np.ndarray
        The edge images (N, 227, 227, 1).
    """
    
    edge_images = []
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobol kernel for x-direction
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # Sobol kernel for y-direction

    for gray_image in X_grayscale:
        # Apply the Sobol kernels

        gray_image = (gray_image / 255.0).astype(np.float32) # Normalize to 0-1

        gradient_x = cv2.filter2D(gray_image, -1, kernel_x)
        gradient_y = cv2.filter2D(gray_image, -1, kernel_y)
        
        # Compute the magnitude of the gradients
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2) * 255.0 # Scale to 0-255
        edge_images.append(gradient_magnitude)
    
    edge_images = np.array(edge_images, dtype=np.uint8)
    edge_images.reshape(-1, 227, 227, 1)

    return edge_images.reshape(-1, 227, 227, 1)


def convert_threshold(X_grayscale: np.ndarray) -> np.ndarray:
    """
    Apply thresholding to the grayscale image according to the Otsu method.
    
    Parameters
    ----------
    X_grayscale : np.ndarray
        The input grayscale images (N, 227, 227, 1).
        
    Returns
    -------
    thresholded_images : np.ndarray
        The thresholded images (N, 227, 227, 1).
    """
    
    thresholded_images = []
    threshold_values = np.arange(256)

    for gray_image in X_grayscale:
        threshold = otsu_threshold(gray_image, threshold_values)
        
        binary_image = gray_image >= threshold
        thresholded_images.append(binary_image)

    return np.array(thresholded_images, dtype=np.uint8)


def otsu_threshold(X_grayscale: np.ndarray, threshold_values: np.ndarray) -> float:
    """
    Apply thresholding to the grayscale image according to the Otsu method.
    
    Parameters
    ----------
    X_grayscale : np.ndarray
        The input grayscale images (227, 227, 1).

    threshold_values : np.ndarray
        The threshold values to consider (256,).
    
    Returns
    -------
    optimal_threshold : float
        The optimal threshold value.
    """

    histogram, bin_edges = np.histogram(X_grayscale, bins=256, range=(0, 255))

    w0 = np.cumsum(histogram)/np.cumsum(histogram)[-1]
    w1 = 1 - w0

    mu = np.sum(histogram * threshold_values)/np.sum(histogram)

    # Computing the mean values of the two classes: background and target
    mu0 = np.cumsum(histogram * threshold_values) / (np.cumsum(histogram) + 1e-12) # Addind small value to avoid division by zero
    mu1 = (mu - mu0*w0) / (w1 + 1e-12) # Addind small value to avoid division by zero

    sigma_b_squared = w0 * (mu0 - mu)**2 + w1 * (mu1 - mu)**2

    optimal_threshold = np.argmax(sigma_b_squared)
    return optimal_threshold
