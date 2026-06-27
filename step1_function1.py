# these are other functions for smoothening

def bilateral_filter1d(data, spatial_window=11, sigma_intensity=0.02):
    """
    1D Bilateral Signal Filter.
    Smooths high-frequency camera noise while preserving sharp structural edges.
    Combines a spatial distance weight with an intensity variance weight.
    """
    n = len(data)
    smoothed = np.copy(data)
    half_window = spatial_window // 2

    # Precompute geometric spatial Gaussian weights based on distance from the center index
    spatial_indices = np.arange(-half_window, half_window + 1)
    spatial_weights = np.exp(-0.5 * (spatial_indices / (spatial_window / 3)) ** 2)

    for i in range(n):
        # Calculate dynamic window boundaries near array edges
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        region = data[start:end]
        current_val = data[i]

        # Clip spatial weight masks to accommodate array boundary truncations
        w_start = half_window - (i - start)
        w_end = half_window + (end - i)
        current_spatial_weights = spatial_weights[w_start:w_end]

        # Compute range weights: Penalizes values that vary greatly from the target data point
        intensity_weights = np.exp(-0.5 * ((region - current_val) / sigma_intensity) ** 2)

        # Combine weights, normalize, and update the signal point
        combined_weights = current_spatial_weights * intensity_weights
        smoothed[i] = np.sum(region * combined_weights) / (np.sum(combined_weights) + 1e-9)
    return smoothed


def top_hat_baseline_suppression(data, window_size=15):
    """
    Morphological Top-Hat Filter.
    Estimates a rolling baseline floor using a grayscale mathematical opening operation,
    then subtracts it to isolate sudden visual changes from slow lighting drift.
    """
    baseline_floor = grey_opening(data, size=window_size)
    cleaned_signal = data - baseline_floor
    return np.clip(cleaned_signal, 0, None)  # Enforce non-negativity across the array
