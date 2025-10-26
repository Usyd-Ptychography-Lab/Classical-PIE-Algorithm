import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import math

# =========================
# 0) Device and global params
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Running on:", device)

# Format and sampling (only for objects and display; PIE itself performs FFT within the patch)
Nx=Ny=500
# L = 2e-3              # Field of view (m), only used for annotation/understanding
# dx = L / Nx

# PIE parameters
patch = 128            # Patch size (suggest covering main lobe + some side lobes) region = patch x patch is the real size.
step  = 32            # Scanning step (≈30-50% of FWHM, ensure 60-80% overlap)
num_iters = 3000
beta = 0.45          # Object update step size
alpha = 1e-4

# =========================
# 1) Gaussian probe (defined within the patch size)
# =========================
def make_gaussian_probe(patch, sigma_px, device="cpu", energy_norm=True):
    """
    Returns a Gaussian complex amplitude probe within the patch size (phase is zero).
    sigma_px: Gaussian standard deviation (pixels)
    energy_norm: If True, ensures sum |P|^2 = 1
    """
    yy, xx = torch.meshgrid(torch.arange(patch, device=device),
                            torch.arange(patch, device=device),
                            indexing='ij')
    cy, cx = (patch-1)/2.0, (patch-1)/2.0
    P = torch.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma_px**2))
    P = P.to(torch.complex64)

    if energy_norm:
        P = P / torch.sqrt(torch.sum(torch.abs(P)**2) + 1e-20)

    return P

# Set Gaussian width (pixels)
# FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.355*sigma
# Let FWHM ≈ 0.5~0.7 of patch size for stability; e.g. 0.6*patch => sigma ≈ 0.6*patch/2.355
# sigma_px = 0.6 * patch / 2.355 # Gaussian standard deviation (pixels) as per formula. Here FWHM is about 0.6 times patch size
sigma_px = 0.5 * patch / 2.355
probe_patch = make_gaussian_probe(patch, sigma_px, device=device, energy_norm=True)

# ===========================================================================================
# 2) Construct object (complex transmission function): Lenna (with both amplitude + phase)
# ===========================================================================================
import torch.nn.functional as F
PAD_PX = 64  # Padding pixels on each side; try 0, 32, 64; larger values yield finer spectral interpolation but slower computation
def pad2d_zeros(x, pad_px):
    # x: [H, W] complex
    if pad_px <= 0:
        return x
    # PyTorch 2D pad: (left, right, top, bottom)
    return F.pad(x, (pad_px, pad_px, pad_px, pad_px), mode='constant', value=0)
def center_crop(x, patch, pad_px):
    # Center crop from padded [H+2p, W+2p] to [patch, patch]
    if pad_px <= 0:
        return x
    return x[pad_px:pad_px+patch, pad_px:pad_px+patch]
def normalize01(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < eps:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# Read Lenna and convert to grayscale
img = Image.open("lenna.jpg").convert("L") # L mode is grayscale Luminance
arr = np.array(img, dtype=np.float32)

# Resize to (Nx, Ny)
arr = resize(arr, (Nx, Ny), anti_aliasing=True, preserve_range=True)
arr = normalize01(arr)  # [0,1]

# ------------------ Amplitude ------------------
# Physical meaning: Darker areas absorb more → Transmission amplitude is lower
# amp_min sets the minimum transmission (to avoid 0 amplitude issues), contrast controls the contrast
amp_min     = 0.2
amp_contrast = 0.8
# Here we choose "brighter = more transparent", while keeping the minimum transmission amp_min
obj_amp = amp_min + amp_contrast * arr               # ∈ [amp_min, amp_min+amp_contrast] ⊂ (0,1]
obj_amp = torch.tensor(obj_amp, dtype=torch.float32, device=device)

# If you want "brighter = more absorbing (closer to a real black-on-white light-blocking target)", you can change to:
# obj_amp = amp_min + amp_contrast * (1.0 - arr)

# ------------------ Phase ------------------
# Use a smoothed version of the grayscale as a slow-varying phase, then map to [-phase_depth, +phase_depth]
# Phase depth (radians), e.g. π means ±π; can change to 0.5*np.pi, etc.
phase_depth = np.pi

# Put grayscale into torch and apply smoothing (average pooling acts as a box low-pass)
t = torch.tensor(arr, dtype=torch.float32, device=device)           # HxW
t4 = t[None, None, ...]                                             # 1x1xHxW
blur_ksz = 17                                                       # Smoothing kernel size (odd)
pad = blur_ksz // 2
t_blur = F.avg_pool2d(F.pad(t4, (pad, pad, pad, pad), mode='reflect'),
                      kernel_size=blur_ksz, stride=1).squeeze(0).squeeze(0)

# Map the smoothed result to [-phase_depth, +phase_depth]
obj_phase = (t_blur - 0.5) * (2.0 * phase_depth)                    # [-phase_depth, +phase_depth]

# (Optional) Add a bit of weak random phase to increase difficulty and make it closer to a real sample
# rand_strength = 0.2   # Phase noise strength (×phase_depth)
# obj_phase = obj_phase + rand_strength * phase_depth * (torch.rand_like(obj_phase) - 0.5) * 2

# ------------------ Complex Transmission Function ------------------
obj_true = obj_amp * torch.exp(1j * obj_phase)                      # complex64

# ===========================================================================================
# 3) Scanning Path (Top-Left Corner Coordinates)
# ===========================================================================================
# Only scan the regions that can fully accommodate the patch
positions = [(x0, y0)
             for x0 in range(0, Nx - patch + 1, step)
             for y0 in range(0, Ny - patch + 1, step)]
print(f"#positions = {len(positions)}")

def crop(t, x0, y0, patch):
    return t[x0:x0+patch, y0:y0+patch]

# ===========================================================================================
# 4) Generate Measurements (Intensity I = |F{ψ}|^2)
# ===========================================================================================
measuredI = []
with torch.no_grad():
    for (x0, y0) in positions:
        Og = crop(obj_true, x0, y0, patch)
        psi = Og * probe_patch
        psi_pad = pad2d_zeros(psi, PAD_PX)
        W   = torch.fft.fftshift(torch.fft.fft2(psi_pad))
        measuredI.append(torch.abs(W)**2)

# ===========================================================================================
# 5) PIE Reconstruction (Update Object Only)
# ===========================================================================================
obj_guess = torch.ones_like(obj_true, dtype=torch.complex64, device=device)

sse_list = []
for it in range(1, num_iters+1):
    total_err = 0.0
    tot_pix   = 0

    for idx, (x0, y0) in enumerate(positions):
        Og = crop(obj_guess, x0, y0, patch)
        # Forward propagation
        psi_g = Og * probe_patch
        psi_gpad = pad2d_zeros(psi_g, PAD_PX)       # [patch+2p, patch+2p]

        Wg    = torch.fft.fftshift(torch.fft.fft2(psi_gpad))

        # Amplitude constraint: replace magnitude with sqrt(I_meas)
        amp_meas = torch.sqrt(measuredI[idx] + 1e-12)
        Wc = amp_meas * torch.exp(1j * torch.angle(Wg))
        
        psi_cpad = torch.fft.ifft2(torch.fft.ifftshift(Wc))       # [patch+2p, patch+2p]
        psi_c    = center_crop(psi_cpad, patch, PAD_PX)

        # PIE Object Update (Only Patch)
        denom = (torch.abs(probe_patch)**2 + alpha)
        Og_new = Og + beta * torch.conj(probe_patch) * (psi_c - psi_g) / denom

        # Write Back to Object
        obj_guess[x0:x0+patch, y0:y0+patch] = Og_new

        # Error (SSE of intensity difference)
        total_err += torch.sum((measuredI[idx] - torch.abs(Wg)**2)**2).item()
        tot_pix   += Wg.numel()

    sse_list.append(total_err / tot_pix)

    if it % 100 == 0:
        print(f"Iter {it}/{num_iters}, SSE={sse_list[-1]:.3e}")

print("Finished.")

# ===========================================================================================
# 6) Results Visualization
# ===========================================================================================
obj_true_np  = obj_true.detach().cpu().numpy()
obj_guess_np = obj_guess.detach().cpu().numpy()

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("True Amplitude")
plt.imshow(np.abs(obj_true_np), cmap="gray", origin="upper")
plt.colorbar()

plt.subplot(2,3,2)
plt.title("Reconstructed Amplitude")
plt.imshow(np.abs(obj_guess_np), cmap="gray", origin="upper")
plt.colorbar()

plt.subplot(2,3,4)
plt.title("True Phase")
plt.imshow(np.angle(obj_true_np), cmap="twilight", origin="upper")
plt.colorbar()

plt.subplot(2,3,5)
plt.title("Reconstructed Phase")
plt.imshow(np.angle(obj_guess_np), cmap="twilight", origin="upper")
plt.colorbar()

plt.subplot(1,3,3)
plt.title("SSE Convergence")
plt.semilogy(sse_list, "b-")
plt.xlabel("Iteration")
plt.ylabel("SSE (log scale)")

plt.tight_layout()
plt.show()
