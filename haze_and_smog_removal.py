import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# PSNR Calculation
def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Local variance map
def compute_local_variance(gray, ksize=5):
    gray = gray.astype(np.float64)
    mean = cv2.boxFilter(gray, -1, (ksize, ksize))
    mean_sq = cv2.boxFilter(gray * gray, -1, (ksize, ksize))
    return mean_sq - mean * mean

# Fixed dark channel
def fixed_dark_channel(img, window_size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    return cv2.erode(min_channel, kernel).astype(np.uint8)

# Adaptive dark channel using variance-based window control
def adaptive_dark_channel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    var_map = compute_local_variance(gray)
    var_norm = cv2.normalize(var_map, None, 0, 1, cv2.NORM_MINMAX)

    min_channel = np.min(img, axis=2)

    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k_med   = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    k_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    dc_small = cv2.erode(min_channel, k_small)
    dc_med   = cv2.erode(min_channel, k_med)
    dc_large = cv2.erode(min_channel, k_large)

    low_th, high_th = 0.1, 0.3
    low_mask  = var_norm < low_th
    med_mask  = (var_norm >= low_th) & (var_norm < high_th)
    high_mask = var_norm >= high_th

    adaptive_dc = np.zeros_like(min_channel)
    adaptive_dc[low_mask]  = dc_small[low_mask]
    adaptive_dc[med_mask]  = dc_med[med_mask]
    adaptive_dc[high_mask] = dc_large[high_mask]

    window_map = np.zeros_like(min_channel)
    window_map[low_mask] = 7
    window_map[med_mask] = 15
    window_map[high_mask] = 25

    return adaptive_dc.astype(np.uint8), window_map

# Atmospheric light
def get_atmospheric_light(img, dark_channel, top_percent=0.001):
    flat_dark = dark_channel.ravel()
    flat_img = img.reshape(-1, 3)
    num_pixels = int(len(flat_dark) * top_percent)
    idx = np.argsort(flat_dark)[-num_pixels:]
    return np.max(flat_img[idx], axis=0)

# Transmission (Fixed)
def estimate_transmission_fixed(img, A, omega=0.95):
    norm = (img.astype(np.float64) / A) * 255
    norm = norm.astype(np.uint8)
    dc = fixed_dark_channel(norm)
    return 1 - omega * (dc / 255.0)

# Transmission (Adaptive)
def estimate_transmission_adaptive(img, A, omega=0.95):
    norm = (img.astype(np.float64) / A) * 255
    norm = norm.astype(np.uint8)
    dc, window_map = adaptive_dark_channel(norm)
    return 1 - omega * (dc / 255.0), dc, window_map

# Guided filter
def guided_filter(I, p, r, eps):
    I = I.astype(np.float64) / 255.0
    p = p.astype(np.float64)

    mean_I  = cv2.boxFilter(I, -1, (r, r))
    mean_p  = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I*p, -1, (r, r))

    cov_Ip = mean_Ip - mean_I * mean_p
    mean_I2 = cv2.boxFilter(I*I, -1, (r, r))
    var_I   = mean_I2 - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))

    return mean_a * I + mean_b

# FULL PIPELINE
def full_dehaze(img_path):

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb); plt.title("Original Image"); plt.axis("off"); plt.show()

    dc_fixed = fixed_dark_channel(img_rgb)
    dc_adaptive, window_map = adaptive_dark_channel(img_rgb)

    plt.figure(figsize=(14,5))
    plt.subplot(1,3,1); plt.imshow(dc_fixed, cmap='gray'); plt.title("Fixed Dark Channel")
    plt.subplot(1,3,2); plt.imshow(dc_adaptive, cmap='gray'); plt.title("Adaptive Dark Channel")
    plt.subplot(1,3,3); plt.imshow(window_map, cmap='jet'); plt.title("Window Size Map")
    plt.show()

    A = get_atmospheric_light(img_rgb, dc_adaptive)
    A_img = np.zeros_like(img_rgb); A_img[:] = A

    plt.imshow(A_img); plt.title("Atmospheric Light"); plt.axis("off"); plt.show()

    t_fixed = estimate_transmission_fixed(img_rgb, A)
    t_adaptive, _, _ = estimate_transmission_adaptive(img_rgb, A)

    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1); plt.imshow(t_fixed, cmap='gray'); plt.title("Transmission (Fixed)")
    plt.subplot(1,2,2); plt.imshow(t_adaptive, cmap='gray'); plt.title("Transmission (Adaptive)")
    plt.show()

    guidance = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
    t_refined = guided_filter(guidance, t_adaptive, r=60, eps=1e-3)

    plt.imshow(t_refined, cmap='gray'); plt.title("Refined Transmission"); plt.axis("off"); plt.show()

    t_final = np.maximum(t_refined, 0.15)[:,:,None]
    J = (img_rgb.astype(np.float64) - A) / t_final + A
    J = np.clip(J, 0, 255).astype(np.uint8)

    plt.imshow(J); plt.title("Dehazed Image"); plt.axis("off"); plt.show()

    hsv = cv2.cvtColor(J, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(2.0,(8,8))
    v2 = clahe.apply(v)
    final = cv2.cvtColor(cv2.merge([h,s,v2]), cv2.COLOR_HSV2RGB)

    plt.imshow(final); plt.title("Final Output (CLAHE)"); plt.axis("off"); plt.show()

    # SSIM
    score = ssim((t_fixed*255).astype(np.uint8),(t_adaptive*255).astype(np.uint8))
    print("\nSSIM (Fixed vs Adaptive Transmission) =", score)

    # PSNR
    psnr_value = calculate_psnr(img_rgb, final)
    print("PSNR (Original vs Final Output) =", psnr_value, "dB\n")

    # Final comparison
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.imshow(img_rgb); plt.title("Original Image"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(final);   plt.title("Final Dehazed + CLAHE"); plt.axis("off")
    plt.show()


# RUN
full_dehaze("test.jpg")
