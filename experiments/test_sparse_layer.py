# test_sparse_layer.py
import math
import os
import random
from dataclasses import dataclass

import torch  # type: ignore
from torch import nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from layers.sparse_expert import SparseExpertLayer 
from layers.layer_config import MLPConfig                

import matplotlib  # type:ignore
matplotlib.use("MacOSX")  # type: ignore . try TkAgg too
import matplotlib.pyplot as plt  # type: ignore

# ---------- utils ----------
torch.set_printoptions(precision=4, sci_mode=False)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def l2_normalize(x, dim: int = -1, eps: float = 1e-12):
    return F.normalize(x, p=2, dim=dim, eps=eps)


# ---------- toy teacher transform ----------
class Teacher(nn.Module):
    """
    T(x) = norm( W2 @ nlin(W1 @ x) ), with fixed 3x3 W1, W2.
    Both input and output lie on S^2.
    """
    def __init__(self, seed: int = 117):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W1 = torch.randn(3, 3, generator=g)
        W1 = l2_normalize(W1, dim=0)
        W2 = torch.randn(3, 3, generator=g)
        W2 = l2_normalize(W2, dim=0)
        W3 = torch.randn(3, 3, generator=g)
        W3 = l2_normalize(W3, dim=0)
        W4 = torch.randn(3, 3, generator=g)
        W4 = l2_normalize(W4, dim=0)
        W5 = torch.randn(3, 3, generator=g)
        W5 = l2_normalize(W5, dim=0)
        self.register_buffer("W1", W1)
        self.register_buffer("W2", W2)
        self.register_buffer("W3", W3)
        self.register_buffer("W4", W4)
        self.register_buffer("W5", W5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return layer 5 and 2 states.
        """
        h1 = x @ self.W1.T  # (n, 3) -> (n, 3)
        h1_prime = l2_normalize(torch.relu(h1) - 0.5 * torch.relu(-h1), dim=1)
        h2 = h1_prime @ self.W2.T  # (n, 3) -> (n, 3)
        h2_prime = l2_normalize(torch.relu(h2) - 0.5 * torch.relu(-h2), dim=1)
        h3 = h2_prime @ self.W3.T  # (n, 3) -> (n, 3)
        h3_prime = l2_normalize(torch.relu(h3) - 0.5 * torch.relu(-h3), dim=1)
        h4 = h3_prime @ self.W4.T  # (n, 3) -> (n, 3)
        h4_prime = l2_normalize(torch.relu(h4) - 0.5 * torch.relu(-h4), dim=1)
        h5 = h4_prime @ self.W5.T  # (n, 3) -> (n, 3)
        h5_prime = l2_normalize(torch.relu(h5) - 0.5 * torch.relu(-h5), dim=1)
        return h5_prime, h2_prime 


# ---------- small wrapper model ----------
class SparseInvertor(nn.Module):
    """
    Learns F(y_teacher) ≈ x using one SparseExpertLayer. Output is L2-normalized.
    """
    def __init__(self, layer_cfg: MLPConfig):
        super().__init__()
        self.layer0 = SparseExpertLayer(layer_cfg)
        self.layer1 = SparseExpertLayer(layer_cfg)

    def forward(self, y: torch.Tensor):
        h0, aux0 = self.layer0(y)           
        h0 = l2_normalize(h0, dim=-1)
        h1, aux1 = self.layer1(h0)
        out = l2_normalize(h1, dim=-1)    # enforce unit norm as final step

        aux = {}
        for k, v in aux0.items():
            aux[f"layer0_{k}"] = v
        for k, v in aux1.items():
            aux[f"layer1_{k}"] = v
        return out, aux

def mse_loss(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

# ---------- data ----------
def sample_unit_sphere(n: int, device: torch.device):
    x = torch.randn(n, 3, device=device)
    return l2_normalize(x, dim=-1)

# ---------- plotting ----------
def make_figure_live(x_eval, y_eval, layer0, layer1):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axs:
        ax.set_xlabel("dim 0"); ax.set_ylabel("dim 1"); ax.set_aspect("equal", adjustable="box")

    # Left: fixed targets (teal), dynamic preds (red)
    axs[0].set_title("Targets (teal) vs Predictions (red)")
    xe2 = x_eval[:, :2].detach().cpu()
    axs[0].scatter(xe2[:, 0], xe2[:, 1], s=8, alpha=0.6, c="#17becf", label="target x")
    red  = axs[0].scatter([], [], s=12, alpha=0.9, c="C3", label="pred x̂")

    # Middle: teacher inputs y (fixed) + layer0 V columns
    axs[1].set_title("Layer 0: y (gray) and V⁰ readers")
    ye2 = y_eval[:, :2].detach().cpu()
    axs[1].scatter(ye2[:, 0], ye2[:, 1], s=6, alpha=0.25, c="0.5", label=None)

    V0 = layer0.V
    D0, m0, b0 = V0.shape
    V0cols2 = V0.reshape(D0, m0 * b0).T[:, :2].detach().cpu()
    v0_pts = axs[1].scatter(V0cols2[:, 0], V0cols2[:, 1], s=18, alpha=0.9, c="C1", label=None)

    # Right: inputs to layer1 (change as layer0 trains) + layer1 V columns
    axs[2].set_title("Layer 1: y¹ (gray) and V¹ readers")
    with torch.no_grad():
        y1_eval, _ = layer0(y_eval)               # residual output of layer0
        y1_eval = l2_normalize(y1_eval, dim=-1)   # match your model's norm step
    y1e2 = y1_eval[:, :2].detach().cpu()
    y1_gray = axs[2].scatter(y1e2[:, 0], y1e2[:, 1], s=6, alpha=0.25, c="0.5", label="y¹ eval")

    V1 = layer1.V
    D1, m1, b1 = V1.shape
    V1cols2 = V1.reshape(D1, m1 * b1).T[:, :2].detach().cpu()
    v1_pts = axs[2].scatter(V1cols2[:, 0], V1cols2[:, 1], s=18, alpha=0.9, c="C2", label="V¹ cols")

    fig.subplots_adjust(left=0.12, right=0.88, wspace=0.30, top=0.85)

    axs[0].legend(loc="center right", bbox_to_anchor=(-0.02, 0.5),
                frameon=True, framealpha=0.9)             # outside left
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15),
                ncol=2, frameon=True, framealpha=0.9)     # above middle
    axs[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                frameon=True, framealpha=0.9)             # outside right


    plt.ion(); plt.show(block=False)
    fig.canvas.draw(); fig.canvas.flush_events()

    # handles we will update live
    artists = {
        "red": red,          # left preds
        "v0_pts": v0_pts,    # middle V0 points
        "y1_gray": y1_gray,  # right gray y1 cloud
        "v1_pts": v1_pts,    # right V1 points
    }
    return fig, axs, artists


@torch.no_grad()
def update_live_plot(fig, axs, artists, x_eval, y_eval, pred_eval, layer0, layer1):
    # left: update predictions
    xp2 = pred_eval[:, :2].detach().cpu()
    artists["red"].set_offsets(xp2)

    # middle: update V0 points
    V0 = layer0.V
    D0, m0, b0 = V0.shape
    V0cols2 = V0.reshape(D0, m0 * b0).T[:, :2].detach().cpu()
    artists["v0_pts"].set_offsets(V0cols2)

    # right: update y1 gray (depends on layer0) and V1 points
    y1_eval, _ = layer0(y_eval)
    y1_eval = l2_normalize(y1_eval, dim=-1)
    y1e2 = y1_eval[:, :2].detach().cpu()
    artists["y1_gray"].set_offsets(y1e2)

    V1 = layer1.V
    D1, m1, b1 = V1.shape
    V1cols2 = V1.reshape(D1, m1 * b1).T[:, :2].detach().cpu()
    artists["v1_pts"].set_offsets(V1cols2)

    fig.canvas.draw(); fig.canvas.flush_events()
    plt.pause(0.01)


# ---------- training loop ----------
@dataclass
class TrainCfg:
    n_train: int = 2048  # temp test
    n_val: int = 1024  # temp test
    batch_size: int = 128
    steps: int = 12000
    log_every: int = 100
    lr: float = 8e-4
    # single coeff applied to layer's aux_loss. range 1e-3 to ~0.5
    aux_coeff: float = 0.         
    plot_every: int = 10
    out_dir: str = "out_sparse"


def run():
    seed = 337
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # teacher and data
    teacher = Teacher(seed=seed).to(device)

    train_cfg = TrainCfg()
    os.makedirs(train_cfg.out_dir, exist_ok=True)

    raw_data = sample_unit_sphere(train_cfg.n_train, device)
    # teacher maps raw_data -> x -> ... -> y
    # student is tasked with learning F(y) ≈ x
    y_train, x_train = teacher(raw_data)

    # TEMP: for visualization, use same val set as train set
    x_val   = x_train[:train_cfg.n_val]
    y_val   = y_train[:train_cfg.n_val]  # (n_val, 3)
    print(f"Printing y_val samples: {y_val[0:5, :]}")

    # layer config (3D, small K for demo)
    # total num params = D * m * b * 2
    layer_cfg = MLPConfig.sparse_default(D=3)
    layer_cfg.D = 3
    layer_cfg.b = 1
    layer_cfg.m = 16
    layer_cfg.k = 4
    layer_cfg.k_f = 0
    layer_cfg.lambda_coeff = 2.
    layer_cfg.coherence_coeff = 0.0

    model = SparseInvertor(layer_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    # fixed eval set for plotting
    x_eval, y_eval = x_val, y_val

    # figure for live updates
    fig, axs, artists = make_figure_live(x_eval, y_eval, model.layer0, model.layer1)

    # training
    model.train()
    for step in range(1, train_cfg.steps + 1):
        # batch
        idx = torch.randint(0, train_cfg.n_train, (train_cfg.batch_size,), device=device)
        yb, xb = y_train[idx], x_train[idx]

        pred, aux = model(yb)
        task = mse_loss(pred, xb)

        # single aux coefficient applied to layer0's aux_loss and layer1's aux_loss
        aux_loss0 = aux.get("layer0_aux_loss", torch.zeros((), device=task.device))
        aux_loss1 = aux.get("layer1_aux_loss", torch.zeros((), device=task.device))
        loss = task + train_cfg.aux_coeff * (aux_loss0 + aux_loss1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % train_cfg.log_every == 0:
            with torch.no_grad():
                model.eval()
                pred_val, aux_val = model(y_val)
                val_task = mse_loss(pred_val, x_val).item()
                val_aux0 = float(aux_val.get("layer0_aux_loss", torch.tensor(0.0)))
                val_aux1 = float(aux_val.get("layer1_aux_loss", torch.tensor(0.0)))
                # expert breakdowns just on layer0
                val_total_energy = aux_val["layer0_total_energy"].item()
                val_energy_per_expert = aux_val["layer0_energy_per_expert"].detach().cpu()  # (m,)
                val_select_rate = aux_val["layer0_select_rate"].detach().cpu()  # (m,)
                model.train()

                print(
                    f"step {step:4d} | train {loss.item():.4f} "
                    f"(task {task.item():.4f} + {train_cfg.aux_coeff}*aux0 {aux_loss0.item():.4f} "
                    f"+ {train_cfg.aux_coeff}*aux1 {aux_loss1.item():.4f}) "
                    f"| val task {val_task:.4f} | val aux0 {val_aux0:.4f} + val aux1 {val_aux1:.4f}\n"
                    f"total energy {val_total_energy:.4f}\n"
                    f"energy per expert {val_energy_per_expert}\n"
                    f"select rate {val_select_rate}\n\n"
                )
        if step % train_cfg.plot_every == 0 or step == 1:
            with torch.no_grad():
                model.eval()
                pred_eval, _ = model(y_eval)
                model.train()
            update_live_plot(fig, axs, artists, x_eval, y_eval, pred_eval, model.layer0, model.layer1)

    with torch.no_grad():
        model.eval()
        pred_eval, _ = model(y_eval)
    
    # final plot
    plt.ioff()
    plt.show()  # blocks until the window is closed
    

if __name__ == "__main__":
    run()
