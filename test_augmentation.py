import cv2
import numpy as np
import os
import random


def add_rain_overlay(
    image,
    rain_lines=200,
    rain_length=(15, 40),
    rain_angle=(-15, 15),
    rain_thickness=(1, 2),
    rain_alpha=0.5,
):
    h, w = image.shape[:2]
    rain_layer = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(rain_lines):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        angle = np.deg2rad(np.random.uniform(*rain_angle))
        length = np.random.randint(*rain_length)
        x2 = int(x1 + length * np.sin(angle))
        y2 = int(y1 + length * np.cos(angle))
        color = (np.random.randint(180, 255),) * 3  # light gray/white
        thickness = np.random.randint(*rain_thickness)
        cv2.line(rain_layer, (x1, y1), (x2, y2), color, thickness)
    # Motion blur
    ksize = (1, 15)
    rain_layer = cv2.blur(rain_layer, ksize)
    # Blend
    return cv2.addWeighted(image, 1, rain_layer, rain_alpha, 0)


def add_dew_drops(image, dew_dir="dew_drops", max_drops=10):
    h, w = image.shape[:2]
    dew_files = [
        os.path.join(dew_dir, f) for f in os.listdir(dew_dir) if f.endswith(".png")
    ]
    output = image.copy()
    for _ in range(random.randint(3, max_drops)):
        dew_img = cv2.imread(random.choice(dew_files), cv2.IMREAD_UNCHANGED)
        if dew_img is None:
            continue
        scale = random.uniform(0.05, 0.15)
        dew_h, dew_w = int(dew_img.shape[0] * scale), int(dew_img.shape[1] * scale)
        dew_img = cv2.resize(dew_img, (dew_w, dew_h), interpolation=cv2.INTER_AREA)
        x = random.randint(0, w - dew_w)
        y = random.randint(0, h - dew_h)
        # Alpha blending
        alpha = dew_img[..., 3:] / 255.0
        for c in range(3):
            output[y : y + dew_h, x : x + dew_w, c] = (
                alpha[..., 0] * dew_img[..., c]
                + (1 - alpha[..., 0]) * output[y : y + dew_h, x : x + dew_w, c]
            )
    return output


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.random.rand(res[0] + 1, res[1] + 1, 2) * 2 - 1
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)

    lin_y = np.linspace(0, res[0], shape[0], endpoint=False)
    lin_x = np.linspace(0, res[1], shape[1], endpoint=False)
    y, x = np.meshgrid(lin_y, lin_x, indexing="ij")
    x0 = x.astype(int)
    y0 = y.astype(int)
    dx = x - x0
    dy = y - y0

    def dot_grid(ix, iy, dx, dy):
        gradient = grid[iy, ix]
        return dx * gradient[..., 0] + dy * gradient[..., 1]

    n00 = dot_grid(x0, y0, dx, dy)
    n10 = dot_grid(x0 + 1, y0, dx - 1, dy)
    n01 = dot_grid(x0, y0 + 1, dx, dy - 1)
    n11 = dot_grid(x0 + 1, y0 + 1, dx - 1, dy - 1)

    u = f(dx)
    v = f(dy)

    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    nxy = nx0 * (1 - v) + nx1 * v

    nxy = (nxy - nxy.min()) / (nxy.max() - nxy.min() + 1e-8)
    return nxy


def add_sun_cloud_shadow(image, shadow_strength=0.5, perlin_res=(8, 8)):
    h, w = image.shape[:2]
    noise = generate_perlin_noise_2d((h, w), perlin_res)
    # Create elliptical mask
    mask = np.zeros((h, w), np.float32)
    center = (
        random.randint(int(w * 0.3), int(w * 0.7)),
        random.randint(int(h * 0.3), int(h * 0.7)),
    )
    axes = (
        random.randint(int(w * 0.3), int(w * 0.5)),
        random.randint(int(h * 0.3), int(h * 0.5)),
    )
    angle = random.uniform(0, 360)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=60, sigmaY=60)
    shadow = 1 - shadow_strength * mask * noise
    shadow = np.clip(shadow, 0.5, 1.0)
    output = (image.astype(np.float32) * shadow[..., None]).astype(np.uint8)
    return output


def add_color_drift(image, hue_shift=20, sat_mult=0.8, val_mult=0.9):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Shift hue (simulate yellow/orange/red)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    # Reduce saturation and value
    hsv[..., 1] *= sat_mult
    hsv[..., 2] *= val_mult
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_edge_decay(
    image, mask=None, erosion_iter=10, decay_color=(42, 42, 165), blur_ksize=7
):
    # If no mask provided, use color thresholding to get leaf mask
    if mask is None:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 40, 40])  # adjust for your leaf color
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    # Erode to get edge
    edge = mask - cv2.erode(mask, None, iterations=erosion_iter)
    edge = cv2.GaussianBlur(edge, (blur_ksize, blur_ksize), 0)
    edge_norm = edge.astype(np.float32) / 255.0
    # Create decay overlay
    decay_layer = np.full_like(image, decay_color, dtype=np.uint8)
    # Blend decay color into edge
    decayed = image.copy().astype(np.float32)
    for c in range(3):
        decayed[..., c] = (
            decayed[..., c] * (1 - edge_norm) + decay_layer[..., c] * edge_norm
        )
    # Optionally add noise
    noise = np.random.normal(0, 10, image.shape).astype(np.float32)
    decayed = np.clip(decayed + edge_norm[..., None] * noise, 0, 255)
    return decayed.astype(np.uint8)


def apply_background_blur(image, blur_ksize=(15, 15)):
    # Step 1: Threshold the leaf in HSV space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])  # adjust as needed
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Step 2: Create inverse mask for background
    mask_inv = cv2.bitwise_not(mask)

    # Step 3: Blur the entire image
    blurred = cv2.GaussianBlur(image, blur_ksize, 0)

    # Step 4: Combine sharp foreground and blurred background
    fg = cv2.bitwise_and(image, image, mask=mask)
    bg = cv2.bitwise_and(blurred, blurred, mask=mask_inv)
    return cv2.add(fg, bg)


def replace_background(image, bg_dir="backgrounds"):
    h, w = image.shape[:2]
    bg_files = [
        os.path.join(bg_dir, f)
        for f in os.listdir(bg_dir)
        if f.endswith((".jpg", ".png"))
    ]
    bg = cv2.imread(random.choice(bg_files))
    if bg is None:
        return image
    bg = cv2.resize(bg, (w, h))

    # Create leaf mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)

    # Extract foreground (leaf) and background
    fg = cv2.bitwise_and(image, image, mask=mask)
    bg_masked = cv2.bitwise_and(bg, bg, mask=mask_inv)
    return cv2.add(fg, bg_masked)


def attention_guided_erasing(
    image, attention_map, p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3)
):
    """
    Debug version: force visible red patch regardless of attention mask.
    """
    if random.random() > p:
        return image

    H, W = attention_map.shape
    area = H * W

    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(*ratio)
        h = int(round(np.sqrt(target_area * aspect_ratio)))
        w = int(round(np.sqrt(target_area / aspect_ratio)))
        if h < H and w < W:
            y = random.randint(0, H - h)
            x = random.randint(0, W - w)
            print(f"Erasing patch size: {h}x{w} at ({y},{x})")

            # Force red patch for visual confirmation
            red_patch = np.full((h, w, 3), (0, 0, 255), dtype=np.uint8)
            image[y : y + h, x : x + w] = red_patch
            break
    return image


# Example usage:
if __name__ == "__main__":
    img = cv2.imread(
        "dataset/Plant_leaf_diseases_dataset/test/Blueberry___healthy/image (12).JPG"
    )
    rain_img = add_rain_overlay(img)
    # dew_img = add_dew_drops(img, dew_dir="dew_drops")
    shadow_img = add_sun_cloud_shadow(img)
    cv2.imwrite(
        "test_augment/test_rain.jpg",
        rain_img,
    )
    # cv2.imwrite("test_augment/test_dew.jpg", dew_img)
    cv2.imwrite(
        "test_augment/test_shadow.jpg",
        shadow_img,
    )
    color_drift_img = add_color_drift(img, hue_shift=30, sat_mult=0.8, val_mult=0.9)
    edge_decay_img = add_edge_decay(img)
    cv2.imwrite(
        "test_augment/test_color_drift.jpg",
        color_drift_img,
    )
    cv2.imwrite(
        "test_augment/test_edge_decay.jpg",
        edge_decay_img,
    )
    blur_img = apply_background_blur(img)
    # domain_random_img = replace_background(img, bg_dir="backgrounds")

    cv2.imwrite("test_augment/test_blur_bg.jpg", blur_img)
    # cv2.imwrite("test_augment/test_domain_random.jpg", domain_random_img)

    # Simulate dummy attention map (center-focused)
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    distance = np.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
    attention_map = 1 - distance / distance.max()  # high attention at center
    attention_map = np.clip(attention_map, 0, 1)

    # Apply attention-guided erasing
    erased_img = attention_guided_erasing(img.copy(), attention_map, p=0)

    cv2.imwrite("test_augment/test_attention_erasing.jpg", erased_img)
