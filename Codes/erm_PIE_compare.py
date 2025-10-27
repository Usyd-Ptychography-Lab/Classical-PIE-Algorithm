import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image
import math
import random
import torch.nn.functional as F

# =========================
# 0) Global config & utils
# =========================
def get_device():
    return torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")

def fix_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def normalize01(arr, eps=1e-12):
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < eps:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def make_gaussian_probe(patch, sigma_px, device="cpu", energy_norm=True):
    yy, xx = torch.meshgrid(torch.arange(patch, device=device),
                            torch.arange(patch, device=device),
                            indexing='ij')
    cy, cx = (patch-1)/2.0, (patch-1)/2.0
    P = torch.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma_px**2))
    P = P.to(torch.complex64)
    if energy_norm:
        P = P / torch.sqrt(torch.sum(torch.abs(P)**2) + 1e-20)
    return P

def crop(t, x0, y0, patch):
    return t[x0:x0+patch, y0:y0+patch]

# =========================
# 1) Scene builder (shared)
# =========================
def build_scene(Nx=500, Ny=500, patch=128, step=32, device=None,
                amp_min=0.2, amp_contrast=0.8, phase_depth=math.pi,
                lenna_path="lenna.jpg"):
    """
    Build ground-truth object, probe_true, probe_guess, scanning positions and measured intensities.
    """
    if device is None:
        device = get_device()

    # ----- True & initial probe -----
    sigma_true = 0.5 * patch / 2.355
    sigma_guess = 0.6 * patch / 2.355
    probe_true = make_gaussian_probe(patch, sigma_true, device=device, energy_norm=True)
    probe_guess = make_gaussian_probe(patch, sigma_guess, device=device, energy_norm=True)

    # ----- Object amplitude & phase -----
    try:
        img = Image.open(lenna_path).convert("L")
        arr = np.array(img, dtype=np.float32)
    except Exception:
        print("[WARN] lenna.jpg not found, using random phase plate instead")
        arr = np.random.rand(Nx, Ny).astype(np.float32)

    arr = resize(arr, (Nx, Ny), anti_aliasing=True, preserve_range=True)
    arr = normalize01(arr)

    obj_amp = amp_min + amp_contrast * arr
    obj_amp = torch.tensor(obj_amp, dtype=torch.float32, device=device)

    _t = torch.tensor(arr, dtype=torch.float32, device=device)
    _t4 = _t[None, None, ...]
    blur_ksz = 17
    pad = blur_ksz // 2
    _t_blur = F.avg_pool2d(F.pad(_t4, (pad, pad, pad, pad), mode='reflect'),
                           kernel_size=blur_ksz, stride=1).squeeze(0).squeeze(0)
    obj_phase = (_t_blur - 0.5) * (2.0 * phase_depth)
    obj_true = obj_amp * torch.exp(1j * obj_phase)

    # ----- Scan positions -----
    positions = [(x0, y0)
                 for x0 in range(0, Nx - patch + 1, step)
                 for y0 in range(0, Ny - patch + 1, step)]
    print(f"#positions = {len(positions)}")

    # ----- Measurements -----
    measuredI = []
    with torch.no_grad():
        for (x0, y0) in positions:
            Og = crop(obj_true, x0, y0, patch)
            psi = Og * probe_true
            W   = torch.fft.fftshift(torch.fft.fft2(psi))
            measuredI.append(torch.abs(W)**2)

    return {
        "obj_true": obj_true,
        "probe_true": probe_true,
        "probe_guess": probe_guess,
        "positions": positions,
        "measuredI": measuredI,
        "device": device,
        "Nx": Nx, "Ny": Ny, "patch": patch, "step": step
    }

# =========================
# 2) ePIE
# =========================
def run_ePIE(scene,
             num_iters=200, beta_o=1.0, beta_p=0.01,
             alpha_o=1e-10, alpha_p=1e-10,
             renorm_probe_each=1):
    device = scene["device"]
    patch = scene["patch"]
    positions = scene["positions"]
    measuredI = scene["measuredI"]
    obj_true = scene["obj_true"]  # not used for updates, just to size-match
    probe_guess = scene["probe_guess"]

    obj_guess  = torch.ones_like(obj_true, dtype=torch.complex64, device=device)
    probe_curr = probe_guess.clone()

    sse_list = []

    for it in range(1, num_iters + 1):
        total_err = 0.0
        tot_pix   = 0
        order = list(range(len(positions)))
        random.shuffle(order)

        for k, idx in enumerate(order):
            x0, y0 = positions[idx]
            Og = crop(obj_guess, x0, y0, patch)
            Pg = probe_curr

            psi_g = Og * Pg
            Wg    = torch.fft.fftshift(torch.fft.fft2(psi_g))
            amp_meas = torch.sqrt(measuredI[idx] + 1e-12)
            Wc = amp_meas * torch.exp(1j * torch.angle(Wg))
            psi_c = torch.fft.ifft2(torch.fft.ifftshift(Wc))

            denom_o = torch.max(torch.abs(Pg)**2) + alpha_o
            Og_new  = Og + beta_o * torch.conj(Pg) * (psi_c - psi_g) / denom_o

            denom_p = torch.max(torch.abs(Og)**2) + alpha_p
            Pg_new  = Pg + beta_p * torch.conj(Og) * (psi_c - psi_g) / denom_p

            obj_guess[x0:x0+patch, y0:y0+patch] = Og_new
            probe_curr = Pg_new

            if renorm_probe_each and ((k + 1) % renorm_probe_each == 0):
                energy = torch.sqrt(torch.sum(torch.abs(probe_curr)**2) + 1e-20)
                probe_curr = probe_curr / energy

            total_err += torch.sum((torch.sqrt(measuredI[idx]) - torch.abs(Wg))**2).item()
            tot_pix   += Wg.numel()

        sse_list.append(total_err / tot_pix)

    return {
        "sse": sse_list,
        "obj_rec": obj_guess.detach().cpu().numpy(),
        "probe_rec": probe_curr.detach().cpu().numpy()
    }

# =========================
# 3) rPIE
# =========================
def run_rPIE(scene,
             num_iters=200, beta_o=1.0, beta_p=0.01,
             alpha_o=1e-10, alpha_p=1e-10,
             gamma=0.5, eps=1e-12,
             renorm_probe_each=1):
    device = scene["device"]
    patch = scene["patch"]
    positions = scene["positions"]
    measuredI = scene["measuredI"]
    obj_true = scene["obj_true"]
    probe_guess = scene["probe_guess"]

    obj_guess  = torch.ones_like(obj_true, dtype=torch.complex64, device=device)
    probe_curr = probe_guess.clone()

    sse_list = []

    for it in range(1, num_iters + 1):
        total_err = 0.0
        tot_pix   = 0
        order = list(range(len(positions)))
        random.shuffle(order)

        for k, idx in enumerate(order):
            x0, y0 = positions[idx]
            Og = crop(obj_guess, x0, y0, patch)
            Pg = probe_curr

            psi_g = Og * Pg
            Wg    = torch.fft.fftshift(torch.fft.fft2(psi_g))
            amp_meas = torch.sqrt(measuredI[idx] + 1e-12)
            Wc = amp_meas * torch.exp(1j * torch.angle(Wg))
            psi_c = torch.fft.ifft2(torch.fft.ifftshift(Wc))
            dpsi  = psi_c - psi_g

            denom_o = (1 - gamma) * torch.max(torch.abs(Pg)**2) + gamma * (torch.abs(Pg)**2) + eps + alpha_o
            denom_p = (1 - gamma) * torch.max(torch.abs(Og)**2) + gamma * (torch.abs(Og)**2) + eps + alpha_p

            Og_new = Og + beta_o * torch.conj(Pg) * dpsi / denom_o
            Pg_new = Pg + beta_p * torch.conj(Og) * dpsi / denom_p

            obj_guess[x0:x0+patch, y0:y0+patch] = Og_new
            probe_curr = Pg_new

            if renorm_probe_each and ((k + 1) % renorm_probe_each == 0):
                energy = torch.sqrt(torch.sum(torch.abs(probe_curr)**2) + 1e-20)
                probe_curr = probe_curr / energy

            total_err += torch.sum((torch.sqrt(measuredI[idx]) - torch.abs(Wg))**2).item()
            tot_pix   += Wg.numel()

        sse_list.append(total_err / tot_pix)

    return {
        "sse": sse_list,
        "obj_rec": obj_guess.detach().cpu().numpy(),
        "probe_rec": probe_curr.detach().cpu().numpy()
    }

# =========================
# 4) mPIE (rPIE + momentum)
# =========================
def run_mPIE(scene,
             num_iters=200, beta_o=1.0, beta_p=0.01,
             alpha_o=1e-10, alpha_p=1e-10,
             gamma=0.5, eps=1e-12,
             mu_o=0.7, mu_p=0.2, use_nesterov=False,
             grad_clip=5.0,
             renorm_probe_each=1):
    device = scene["device"]
    patch = scene["patch"]
    positions = scene["positions"]
    measuredI = scene["measuredI"]
    obj_true = scene["obj_true"]
    probe_guess = scene["probe_guess"]

    obj_guess  = torch.ones_like(obj_true, dtype=torch.complex64, device=device)
    probe_curr = probe_guess.clone()

    v_obj   = torch.zeros_like(obj_guess,  dtype=torch.complex64)
    v_probe = torch.zeros_like(probe_curr, dtype=torch.complex64)

    sse_list = []

    for it in range(1, num_iters + 1):
        total_err = 0.0
        tot_pix   = 0
        order = list(range(len(positions)))
        random.shuffle(order)

        for k, idx in enumerate(order):
            x0, y0 = positions[idx]

            if use_nesterov:
                Og_base = obj_guess[x0:x0+patch, y0:y0+patch]
                v_slice = v_obj[x0:x0+patch, y0:y0+patch]
                Og = Og_base + mu_o * v_slice
                Pg = probe_curr + mu_p * v_probe
            else:
                Og = obj_guess[x0:x0+patch, y0:y0+patch]
                Pg = probe_curr

            psi_g = Og * Pg
            Wg    = torch.fft.fftshift(torch.fft.fft2(psi_g))
            amp_meas = torch.sqrt(measuredI[idx] + 1e-12)
            Wc = amp_meas * torch.exp(1j * torch.angle(Wg))
            psi_c = torch.fft.ifft2(torch.fft.ifftshift(Wc))
            dpsi  = psi_c - psi_g

            denom_o = (1 - gamma) * torch.max(torch.abs(Pg)**2) + gamma * (torch.abs(Pg)**2) + eps + alpha_o
            denom_p = (1 - gamma) * torch.max(torch.abs(Og)**2) + gamma * (torch.abs(Og)**2) + eps + alpha_p

            g_o = torch.conj(Pg) * dpsi / denom_o
            g_p = torch.conj(Og) * dpsi / denom_p

            if grad_clip and grad_clip > 0:
                go_norm = torch.linalg.vector_norm(g_o)
                if go_norm.real > grad_clip:
                    g_o = g_o * (grad_clip / (go_norm.real + 1e-12))
                gp_norm = torch.linalg.vector_norm(g_p)
                if gp_norm.real > grad_clip:
                    g_p = g_p * (grad_clip / (gp_norm.real + 1e-12))

            v_slice = v_obj[x0:x0+patch, y0:y0+patch]
            v_slice = mu_o * v_slice + beta_o * g_o
            v_obj[x0:x0+patch, y0:y0+patch] = v_slice
            Og_new = Og + v_slice
            obj_guess[x0:x0+patch, y0:y0+patch] = Og_new

            v_probe = mu_p * v_probe + beta_p * g_p
            probe_curr = Pg + v_probe

            if renorm_probe_each and ((k + 1) % renorm_probe_each == 0):
                energy = torch.sqrt(torch.sum(torch.abs(probe_curr)**2) + 1e-20)
                probe_curr = probe_curr / energy

            total_err += torch.sum((torch.sqrt(measuredI[idx]) - torch.abs(Wg))**2).item()
            tot_pix   += Wg.numel()

        sse_list.append(total_err / tot_pix)

    return {
        "sse": sse_list,
        "obj_rec": obj_guess.detach().cpu().numpy(),
        "probe_rec": probe_curr.detach().cpu().numpy()
    }

# =========================
# 5) Plot helper
# =========================
def plot_first200(e_err, r_err, m_err, metric_name="SSE / Error", save_path=None, use_logy=True):
    import numpy as np
    import matplotlib.pyplot as plt

    e = np.array(e_err)[:200]
    r = np.array(r_err)[:200]
    m = np.array(m_err)[:200]

    x_e = np.arange(1, len(e) + 1)
    x_r = np.arange(1, len(r) + 1)
    x_m = np.arange(1, len(m) + 1)

    plt.figure()
    plt.plot(x_e, e, label="ePIE")
    plt.plot(x_r, r, label="rPIE")
    plt.plot(x_m, m, label="mPIE")
    if use_logy:
        plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.title("First 1000 Iterations: ePIE vs rPIE vs mPIE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
    plt.show()

# =========================
# 6) Main (run & compare)
# =========================
if __name__ == "__main__":
    device = get_device()
    print("Running on:", device)
    fix_seeds(0)

    # ---- Shared scene ----
    scene = build_scene(Nx=500, Ny=500, patch=128, step=32, device=device)

    # ---- Hyper-params (aligned to your originals) ----
    common = dict(num_iters=200, beta_o=1.0, beta_p=0.01, alpha_o=1e-10, alpha_p=1e-10)
    rpie_extra = dict(gamma=0.5, eps=1e-12)
    mpie_extra = dict(gamma=0.5, eps=1e-12, mu_o=0.7, mu_p=0.2, use_nesterov=False, grad_clip=5.0)

    # ---- Run three algorithms ----
    out_e = run_ePIE(scene, **common)
    out_r = run_rPIE(scene, **common, **rpie_extra)
    out_m = run_mPIE(scene, **common, **mpie_extra)

    # ---- Plot first 200 on one figure ----
    plot_first200(out_e["sse"], out_r["sse"], out_m["sse"], metric_name="SSE", save_path=None, use_logy=True)

    # (可选) 如果要看重建结果，可在此添加可视化
    # obj_true_np = scene["obj_true"].detach().cpu().numpy()
    # ... your visualization code ...
