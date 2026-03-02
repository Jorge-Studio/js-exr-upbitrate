"""
Fractal mathematics utilities for bit-depth expansion.

Provides:
- Local Fractal Dimension (LFD) via box-counting
- Fractal Brownian Motion (fBm) noise generation
- Hermite spline interpolation for smooth gradient regions
- Rational Fractal Cubic (RFC) spline for textured regions
- Blue noise and TPDF dither generators
"""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def compute_local_fractal_dimension(image_gray, patch_size=7, scales=None):
    """
    Compute per-pixel Local Fractal Dimension using differential box-counting.

    For each patch, measures how complexity changes across scales.
    Returns a map where:
      ~1.0 = smooth gradient (sky, solid color)
      ~1.5 = organic texture (skin, cloth)
      ~2.0 = complex detail (hair, foliage edge)
    """
    if scales is None:
        scales = [2, 3, 4, 6]

    h, w = image_gray.shape
    half = patch_size // 2

    img = (image_gray * 255).astype(np.float64) if image_gray.max() <= 1.0 else image_gray.astype(np.float64)

    lfd_map = np.ones((h, w), dtype=np.float64) * 1.5

    pad = max(scales) + half
    img_padded = np.pad(img, pad, mode='reflect')

    log_counts = []
    log_scales = []

    for s in scales:
        block_size = s
        count_map = np.zeros((h, w), dtype=np.float64)

        for dy in range(-half, half + 1, block_size):
            for dx in range(-half, half + 1, block_size):
                y_start = pad + dy
                x_start = pad + dx

                patch = img_padded[y_start:y_start + h, x_start:x_start + w]
                local_max = uniform_filter(patch, size=block_size, mode='reflect')
                local_min = uniform_filter(
                    -patch, size=block_size, mode='reflect'
                )
                local_range = local_max + local_min
                count_map += np.maximum(local_range / block_size, 1.0)

        count_map = np.maximum(count_map, 1.0)
        log_counts.append(np.log(count_map))
        log_scales.append(np.log(1.0 / s))

    log_counts = np.array(log_counts)
    log_scales = np.array(log_scales)

    # Linear regression per pixel: LFD = slope of log(count) vs log(1/scale)
    n = len(scales)
    sum_x = log_scales.sum()
    sum_x2 = (log_scales ** 2).sum()
    sum_y = log_counts.sum(axis=0)
    sum_xy = (log_counts * log_scales[:, None, None]).sum(axis=0)

    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) > 1e-10:
        lfd_map = (n * sum_xy - sum_x * sum_y) / denom

    lfd_map = np.clip(lfd_map, 1.0, 2.5)

    # Smooth the LFD map to avoid per-pixel noise
    lfd_map = gaussian_filter(lfd_map, sigma=2.0)

    return lfd_map.astype(np.float32)


def fractal_brownian_motion(shape, octaves=4, persistence=0.5, lacunarity=2.0,
                            seed=42):
    """
    Generate a 2D Fractal Brownian Motion (fBm) noise field.

    Returns values in [-1, 1] range with fractal self-similarity.
    Higher octaves = more fine detail. Persistence controls amplitude falloff.
    """
    rng = np.random.RandomState(seed)
    h, w = shape
    result = np.zeros((h, w), dtype=np.float64)

    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        # Generate smooth noise at this frequency via interpolated random grid
        grid_h = max(2, int(h * frequency / max(h, w)))
        grid_w = max(2, int(w * frequency / max(h, w)))

        noise_grid = rng.randn(grid_h + 2, grid_w + 2).astype(np.float64)

        # Bicubic upscale to full resolution
        from scipy.ndimage import zoom
        scale_y = h / noise_grid.shape[0]
        scale_x = w / noise_grid.shape[1]
        noise_full = zoom(noise_grid, (scale_y, scale_x), order=3)

        # Crop to exact size
        noise_full = noise_full[:h, :w]

        result += noise_full * amplitude
        max_amplitude += amplitude

        amplitude *= persistence
        frequency *= lacunarity

    if max_amplitude > 0:
        result /= max_amplitude

    return np.clip(result, -1.0, 1.0).astype(np.float32)


def hermite_interpolate_neighbors(image, channel=None):
    """
    Compute sub-pixel offset for each pixel based on Hermite spline
    interpolation from its 4 neighbors.

    Returns an offset in [-0.5, 0.5] range indicating where this pixel
    sits within its quantization bin relative to the local gradient.
    """
    if channel is not None:
        img = image[:, :, channel].astype(np.float64)
    elif len(image.shape) == 3:
        img = np.mean(image.astype(np.float64), axis=2)
    else:
        img = image.astype(np.float64)

    # Compute tangents from neighbors (central differences)
    # Shift in all 4 directions
    up = np.roll(img, -1, axis=0)
    down = np.roll(img, 1, axis=0)
    left = np.roll(img, -1, axis=1)
    right = np.roll(img, 1, axis=1)

    # Gradient magnitude and direction
    grad_y = (up - down) / 2.0
    grad_x = (left - right) / 2.0

    # Hermite basis: for t=0.5 (midpoint), h00=0.5, h10=0.125, h01=0.5, h11=-0.125
    # The offset tells us where in the quantization bin this pixel should sit
    # based on the local gradient direction
    offset_y = grad_y * 0.25
    offset_x = grad_x * 0.25

    # Combine into a single sub-pixel offset
    offset = np.sqrt(offset_x ** 2 + offset_y ** 2) * np.sign(grad_y + grad_x)

    # Normalize to [-0.5, 0.5]
    max_val = np.abs(offset).max()
    if max_val > 0:
        offset = offset / max_val * 0.5

    return offset.astype(np.float32)


def rational_fractal_cubic_spline(x, alpha=0.5, d_fractal=1.5):
    """
    Rational Fractal Cubic (RFC) spline interpolation.

    Blends standard cubic interpolation with fractal perturbation controlled
    by the local fractal dimension. Higher d_fractal = more irregular fill.

    x: input values in [0, 1] (position within quantization bin)
    alpha: scaling factor for fractal component
    d_fractal: local fractal dimension (1.0 = smooth, 2.0 = rough)
    """
    # Standard cubic Hermite basis
    t = x
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h01 = -2 * t ** 3 + 3 * t ** 2

    # Fractal perturbation: IFS-inspired self-affine transform
    fractal_weight = np.clip((d_fractal - 1.0) / 1.0, 0, 1) * alpha
    perturbation = np.sin(t * np.pi * (2 ** d_fractal)) * fractal_weight * 0.1

    result = h00 * 0.0 + h01 * 1.0 + perturbation

    return np.clip(result, 0, 1).astype(np.float32)


def generate_blue_noise(shape, seed=42):
    """
    Generate a blue noise dither pattern using void-and-cluster method
    (simplified). Blue noise has energy concentrated at high frequencies,
    making it perceptually invisible at normal viewing distances.
    """
    rng = np.random.RandomState(seed)
    h, w = shape

    # Start with white noise, then iteratively push toward blue spectrum
    noise = rng.rand(h, w).astype(np.float64)

    # Apply high-pass filtering to shift energy to high frequencies
    for _ in range(3):
        low_freq = gaussian_filter(noise, sigma=1.5)
        noise = noise - low_freq * 0.5 + 0.25

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)

    return noise.astype(np.float32)


def generate_tpdf_dither(shape, seed=42):
    """
    Generate Triangular Probability Density Function dither.
    Sum of two uniform random variables -> triangular distribution.
    Industry standard for bit-depth reduction/expansion.
    """
    rng = np.random.RandomState(seed)
    h, w = shape

    u1 = rng.rand(h, w).astype(np.float64)
    u2 = rng.rand(h, w).astype(np.float64)

    # TPDF: sum of two uniform -> triangular in [-1, 1]
    tpdf = u1 - u2

    return tpdf.astype(np.float32)


def sobel_gradient(image_gray):
    """Compute Sobel gradients. Returns (grad_x, grad_y) both as float32."""
    if len(image_gray.shape) == 3:
        gray = np.mean(image_gray, axis=2)
    else:
        gray = image_gray

    gray = gray.astype(np.float64)

    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / 8.0

    from scipy.ndimage import convolve
    grad_x = convolve(gray, kernel_x, mode='reflect')
    grad_y = convolve(gray, kernel_y, mode='reflect')

    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def temporal_coherent_seed(base_seed, frame_index, coherence=0.8):
    """
    Generate a seed that varies smoothly across frames for temporal coherence.
    coherence=1.0 means same seed every frame (static noise).
    coherence=0.0 means fully random per frame.
    """
    if coherence >= 0.99:
        return base_seed

    # Blend between static and per-frame seeds
    static_component = base_seed
    temporal_component = base_seed + frame_index * 7919  # large prime

    # Use coherence to determine how many octaves share the same seed
    return int(static_component * coherence + temporal_component * (1 - coherence))


def laplacian_pyramid(image, levels=5):
    """
    Decompose image into a Laplacian pyramid (frequency bands).
    Returns list of arrays from lowest to highest frequency.
    """
    pyramid = []
    current = image.astype(np.float64)

    for i in range(levels - 1):
        blurred = gaussian_filter(current, sigma=2 ** (levels - 1 - i) * 0.5)
        detail = current - blurred
        pyramid.append(detail.astype(np.float32))
        current = blurred

    # Residual (lowest frequency)
    pyramid.append(current.astype(np.float32))

    return pyramid


def reconstruct_from_pyramid(pyramid):
    """Reconstruct image from Laplacian pyramid."""
    result = pyramid[-1].astype(np.float64)
    for i in range(len(pyramid) - 2, -1, -1):
        result = result + pyramid[i].astype(np.float64)
    return result.astype(np.float32)
