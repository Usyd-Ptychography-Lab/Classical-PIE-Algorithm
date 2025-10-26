import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import math
import random
import torch.nn.functional as F

# =========================
# 0) Device and global params
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Format and sampling (only for objects and display; ePIE itself performs FFT within the patch)
Nx = Ny = 500

# ePIE parameters (similar to your PIE, with additional probe update related parameters)
patch = 128          # Patch size (region = patch x patch)
step  = 32           # Scanning step (≈30-50% of FWHM, ensure 60-80% overlap)
num_iters = 3000     # ePIE usually random access + faster convergence, can be reduced
beta_o = 1           # Object update step size (similar to your beta)
beta_p = 0.01        # Probe update step size (slightly smaller than object for stability)
alpha_o = 1e-10      # Object update stability term (prevent division by 0)
alpha_p = 1e-10      # Probe update stability term
renorm_probe_each = 1  # Every how many steps to renormalize probe energy (1 means renormalize after each patch)

gamma = 0.5  # Optional: Probe energy renormalization factor (if not wanting to renormalize every time, can set to 1.0)
eps = 1e-12  # Prevent division by zero
# =========================
# 1) Gaussian Probe (defined within the patch size)
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
sigma_px = 0.5 * patch / 2.355
probe_true = make_gaussian_probe(patch, sigma_px, device=device, energy_norm=True)

# Initial probe guess (can be slightly off from true value; here example is a slightly wider Gaussian)
sigma_guess = 0.6 * patch / 2.355
probe_guess = make_gaussian_probe(patch, sigma_guess, device=device, energy_norm=True)

# ===========================================================================================
# 2) Construct object (complex transmission function): Lenna (with both amplitude + phase)
# ===========================================================================================

def normalize01(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < eps:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# Read Lenna and convert to grayscale
try:
    img = Image.open("lenna.jpg").convert("L")
except Exception:
    print("[WARN] lenna.jpg not found, using random phase plate instead")
    arr = np.random.rand(Nx, Ny).astype(np.float32)
else:
    arr = np.array(img, dtype=np.float32)

# Resize to (Nx, Ny)
arr = resize(arr, (Nx, Ny), anti_aliasing=True, preserve_range=True)
arr = normalize01(arr)  # [0,1]

# Amplitude: map grayscale to [amp_min, amp_min + amp_contrast]
amp_min      = 0.2
amp_contrast = 0.8
obj_amp = amp_min + amp_contrast * arr
obj_amp = torch.tensor(obj_amp, dtype=torch.float32, device=device)

# Phase: Use a smoothed version of the grayscale to map to [-phase_depth, +phase_depth]
phase_depth = math.pi
_t = torch.tensor(arr, dtype=torch.float32, device=device)
_t4 = _t[None, None, ...]
blur_ksz = 17
pad = blur_ksz // 2
_t_blur = F.avg_pool2d(F.pad(_t4, (pad, pad, pad, pad), mode='reflect'), kernel_size=blur_ksz, stride=1).squeeze(0).squeeze(0)
obj_phase = (_t_blur - 0.5) * (2.0 * phase_depth)

# Complex transmission function (ground truth)
obj_true = obj_amp * torch.exp(1j * obj_phase)

# ===========================================================================================
# 3) Scanning Path (Top-Left Corner Coordinates)
# ===========================================================================================
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
        psi = Og * probe_true
        W   = torch.fft.fftshift(torch.fft.fft2(psi))
        measuredI.append(torch.abs(W)**2)

# ===========================================================================================
# 5) ePIE engine (Joint Update of Object + Probe)
# ===========================================================================================
obj_guess  = torch.ones_like(obj_true, dtype=torch.complex64, device=device)
probe_curr = probe_guess.clone()

sse_list = []

for it in range(1, num_iters + 1):
    total_err = 0.0
    tot_pix   = 0

    # ePIE common practice: randomly access scanning positions (shuffle) in each iteration
    order = list(range(len(positions)))
    random.shuffle(order)

    for k, idx in enumerate(order):
        x0, y0 = positions[idx]

        # Current patch object slice + current probe
        Og = crop(obj_guess, x0, y0, patch)
        Pg = probe_curr  # Single probe case (if multi-probe, can index by frame)

        # Forward propagation
        psi_g = Og * Pg
        Wg    = torch.fft.fftshift(torch.fft.fft2(psi_g))

        # Amplitude constraint (replace amplitude with sqrt(I_meas))
        amp_meas = torch.sqrt(measuredI[idx] + 1e-12)
        Wc = amp_meas * torch.exp(1j * torch.angle(Wg))
        psi_c = torch.fft.ifft2(torch.fft.ifftshift(Wc))

        dpsi = psi_c - psi_g

        # rPIE denominator (mix max and per-pixel)
        denom_o = (1 - gamma) * torch.max(torch.abs(Pg)**2) + gamma * (torch.abs(Pg)**2) + eps + alpha_o
        denom_p = (1 - gamma) * torch.max(torch.abs(Og)**2) + gamma * (torch.abs(Og)**2) + eps + alpha_p

        # rPIE object update
        Og_new = Og + beta_o * torch.conj(Pg) * dpsi / denom_o
        obj_guess[x0:x0+patch, y0:y0+patch] = Og_new

        # rPIE probe update
        Pg_new = Pg + beta_p * torch.conj(Og) * dpsi / denom_p
        probe_curr = Pg_new  # If it oscillates easily, use EMA: probe_curr = (1-eta)*probe_curr + eta*Pg_new

        # Optional: Keep probe energy normalized to avoid divergence
        if renorm_probe_each and ((k + 1) % renorm_probe_each == 0):
            energy = torch.sqrt(torch.sum(torch.abs(probe_curr)**2) + 1e-20)
            probe_curr = probe_curr / energy

        # Error (SSE of intensity difference)
        total_err += torch.sum((torch.sqrt(measuredI[idx]) - torch.abs(Wg))**2).item()

        tot_pix   += Wg.numel()

    sse_list.append(total_err / tot_pix)

    if it % 100 == 0:
        print(f"Iter {it}/{num_iters}, SSE={sse_list[-1]:.3e}")

print("Finished.")

# ===========================================================================================
# 6) Results Visualization
# ===========================================================================================
obj_true_np   = obj_true.detach().cpu().numpy()
obj_guess_np  = obj_guess.detach().cpu().numpy()
probe_true_np = probe_true.detach().cpu().numpy()
probe_rec_np  = probe_curr.detach().cpu().numpy()

plt.figure(figsize=(13,9))

plt.subplot(3,3,1)
plt.title("True Amplitude")
plt.imshow(np.abs(obj_true_np), cmap="gray", origin="upper")
plt.colorbar()

plt.subplot(3,3,2)
plt.title("Reconstructed Amplitude (ePIE)")
plt.imshow(np.abs(obj_guess_np), cmap="gray", origin="upper")
plt.colorbar()

plt.subplot(3,3,4)
plt.title("True Phase")
plt.imshow(np.angle(obj_true_np), cmap="twilight", origin="upper")
plt.colorbar()

plt.subplot(3,3,5)
plt.title("Reconstructed Phase (ePIE)")
plt.imshow(np.angle(obj_guess_np), cmap="twilight", origin="upper")
plt.colorbar()

plt.subplot(1,3,3)
plt.title("SSE Convergence")
plt.semilogy(sse_list, "b-")
plt.xlabel("Iteration (shuffled scans)")
plt.ylabel("SSE (log scale)")

# Probe comparison
plt.subplot(3,3,7)
plt.title("True Probe |P|")
plt.imshow(np.abs(probe_true_np), cmap="gray", origin="upper")
plt.colorbar()

plt.subplot(3,3,8)
plt.title("Reconstructed Probe |P| (ePIE)")
plt.imshow(np.abs(probe_rec_np), cmap="gray", origin="upper")
plt.colorbar()

plt.tight_layout()
plt.show()
