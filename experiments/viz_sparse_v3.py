import math
from multiprocessing import Value
import numpy as np
import torch  # type: ignore
from torch import nn  # type: ignore
import torch.nn.functional as F  # type: ignore

import matplotlib  # type: ignore
matplotlib.use("MacOSX")  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from layers.sparse_expert_v3 import SparseExpertV3
from layers.layer_config import MLPConfig
from layers.layer_utils import AUX_LOSS_SUFFIX


# ----------------------------
# Utilities: geometry and grids
# ----------------------------
def unit_disk_grid(res: int):
    xs = np.linspace(-1.0, 1.0, res)
    ys = np.linspace(-1.0, 1.0, res)
    X, Y = np.meshgrid(xs, ys)
    mask = (X**2 + Y**2) <= 1.0
    return X, Y, mask


def lift_to_upper_hemisphere(X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    R2 = X**2 + Y**2
    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(np.maximum(1.0 - R2[mask], 0.0))
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=0)  # (3, N)
    return pts


def labels_to_grid(labels: np.ndarray, X: np.ndarray, mask: np.ndarray, fill_value: int = -1) -> np.ndarray:
    out = np.full(X.shape, fill_value=fill_value, dtype=int)
    out[mask] = labels
    return out


def normalize_cols_np(pts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(pts, axis=0, keepdims=True)
    return pts / np.maximum(norms, eps)


@torch.no_grad()
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


# ----------------------------
# Ground truth classifier (fixed, random MLP)
# ----------------------------
class GroundTruthClassifier(nn.Module):
    """
    Fixed small MLP mapping R^D (unit norm, D=3) -> logits over J classes.
    Weights are frozen after init to serve as the teacher labeling function.
    """
    def __init__(self, D: int, J: int, hidden_dim: int = 32, num_hidden_layers: int = 1, seed: int = 1234):
        super().__init__()
        assert num_hidden_layers >= 0, "num_hidden_layers must be >= 0"
        self.D = D
        self.J = J
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        g = torch.Generator().manual_seed(seed)

        layers: list[nn.Linear] = []
        if num_hidden_layers == 0:
            # Direct linear classifier D -> J
            lin = nn.Linear(D, J, bias=True)
            with torch.no_grad():
                w = torch.randn(lin.weight.shape, generator=g, device=lin.weight.device, dtype=lin.weight.dtype)
                w = F.normalize(w, p=2, dim=1)
                lin.weight.copy_(w)
                lin.bias.copy_(torch.randn(lin.bias.shape, generator=g, device=lin.bias.device, dtype=lin.bias.dtype))
            for p in lin.parameters():
                p.requires_grad = False
            layers.append(lin)
        else:
            # First hidden
            lin_in = nn.Linear(D, hidden_dim, bias=True)
            # Middle hidden layers (hidden -> hidden)
            hidden_layers = [nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(max(0, num_hidden_layers - 1))]
            # Output layer
            lin_out = nn.Linear(hidden_dim, J, bias=True)

            with torch.no_grad():
                # Normalize columns for stability (equivalently rows of Linear weight)
                def init_and_norm(linear: nn.Linear):
                    w = torch.randn(linear.weight.shape, generator=g, device=linear.weight.device, dtype=linear.weight.dtype)
                    # normalize over input dim (dim=1 of (out,in))
                    w = F.normalize(w, p=2, dim=1)
                    linear.weight.copy_(w)
                    linear.bias.copy_(torch.randn(linear.bias.shape, generator=g, device=linear.bias.device, dtype=linear.bias.dtype))

                init_and_norm(lin_in)
                for hl in hidden_layers:
                    init_and_norm(hl)
                init_and_norm(lin_out)

            # Freeze
            for p in list(lin_in.parameters()) + [p for hl in hidden_layers for p in hl.parameters()] + list(lin_out.parameters()):
                p.requires_grad = False

            # Register as modules
            layers.append(lin_in)
            layers.extend(hidden_layers)
            layers.append(lin_out)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_hidden_layers == 0:
            return self.layers[0](x)
        h = x
        # apply tanh after each hidden layer, before final output
        for idx, layer in enumerate(self.layers):
            if idx < self.num_hidden_layers:
                h = torch.tanh(layer(h))
            else:
                # final layer to logits
                h = layer(h)
        return h


# ----------------------------
# Model: L-layer SparseExpertV3 stack + linear head
# ----------------------------
class SparseStack(nn.Module):
    """
    x in R^D (unit norm) -> L times SparseExpertV3 -> linear head to J classes
    Each layer output is re-normalized inside SparseExpertV3.
    """
    def __init__(
        self,
        D: int,
        J: int,
        L: int,
        m: int,
        b: int,
        k: int,
        lambda_coeff: float = 1.0,
        selection_relu: bool = False,
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.D = D
        self.J = J
        self.L = L
        self.dtype = dtype
        self.layers = nn.ModuleList()
        for _ in range(L):
            cfg = MLPConfig.sparse_default(D=D)
            cfg.b = b
            cfg.m = m
            cfg.k = k
            cfg.lambda_coeff = lambda_coeff
            cfg.selection_relu = selection_relu
            cfg.norm_eps = eps
            layer = SparseExpertV3(cfg)
            self.layers.append(layer)
        self.head = nn.Linear(D, J, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux_total: dict[str, torch.Tensor] = {}
        h = x
        for li, layer in enumerate(self.layers):
            h, aux = layer(h)
            # Aggregate per-layer aux (sum losses and keep metrics with suffix)
            for k, v in aux.items():
                key = f"layer{li}_{k}"
                aux_total[key] = v
        logits = self.head(h)
        return logits, aux_total


# ----------------------------
# Data sampling
# ----------------------------
@torch.no_grad()
def sample_upper_hemisphere(n: int, D: int, device: torch.device) -> torch.Tensor:
    x = torch.randn(n, D, device=device)
    x = l2_normalize(x, dim=-1)
    # enforce upper hemisphere z >= 0 (assumes D >= 3)
    if D >= 3:
        x[:, 2] = x[:, 2].abs()
        x = l2_normalize(x, dim=-1)
    return x


# ----------------------------
# Visualization helpers
# ----------------------------
def make_boundary_figures(X: np.ndarray, mask: np.ndarray, J: int):
    # Ground truth and model prediction side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        theta = np.linspace(0, 2 * math.pi, 512)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=1.0, color="k", alpha=0.8)

    axs[0].set_title("Ground truth classes")
    axs[1].set_title("Model-predicted classes")

    blank = np.full_like(X, fill_value=-1, dtype=int)
    im_gt = axs[0].imshow(
        blank, origin="lower", extent=[-1, 1, -1, 1], interpolation="nearest", vmin=-0.5, vmax=J - 0.5
    )
    im_model = axs[1].imshow(
        blank, origin="lower", extent=[-1, 1, -1, 1], interpolation="nearest", vmin=-0.5, vmax=J - 0.5
    )
    # Leave room on the right for a non-overlapping colorbar
    fig.subplots_adjust(right=0.86)
    cb = fig.colorbar(im_gt, ax=axs, fraction=0.035, pad=0.03)
    cb.set_ticks(range(J))
    # Tight layout within the reserved rectangle to avoid overlap with colorbar
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
    plt.ion()
    plt.show(block=False)
    fig.canvas.draw(); fig.canvas.flush_events()
    return fig, axs, im_gt, im_model

def make_model_views_figure(J: int, num_views: int, extent=(-1, 1, -1, 1)):
    """
    Create a figure with num_views panels for dynamic model views (no GT).
    Returns (fig, axs, ims) where ims is a list of AxesImage handles to update.
    """
    ncols = num_views
    fig_w = 4.2 * ncols
    fig_h = 4.0
    fig, axs = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    axs = np.atleast_1d(axs).reshape(1, ncols)[0]

    ims = []
    vmin, vmax = -0.5, J - 0.5
    for i in range(ncols):
        ax = axs[i]
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        theta = np.linspace(0, 2 * math.pi, 512)
        ax.plot(np.cos(theta), np.sin(theta), linewidth=1.0, color="k", alpha=0.8)
        blank = np.zeros((int((extent[3]-extent[2])*100), int((extent[1]-extent[0])*100)), dtype=int)
        im = ax.imshow(blank, origin="lower", extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_title(f"head ∘ layer^{i}(x)")

    fig.subplots_adjust(right=0.90)
    cb = fig.colorbar(ims[0], ax=axs, fraction=0.03, pad=0.02)
    cb.set_ticks(range(J))
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    plt.ion(); plt.show(block=False)
    fig.canvas.draw(); fig.canvas.flush_events()
    return fig, axs, ims

def make_unified_views_figure(J: int, num_layers: int, extent=(-1, 1, -1, 1), max_cols: int = 4):
    """
    Create a single figure laid out in multiple rows with GT first, followed by (L+1) model views:
      panels = [GT, head∘layer^0(x), head∘layer^1(x), ..., head∘layer^L(x)]
    Panels are arranged in a grid with up to max_cols per row.
    Returns (fig, axs, ims) where:
      - ims[0] is GT
      - ims[1:] are model views in order
    """
    total = (num_layers + 1) + 1  # L+1 views + 1 GT
    ncols = min(max_cols, int(math.ceil(math.sqrt(total))))
    nrows = int(math.ceil(total / float(ncols)))
    fig_w = 4.0 * ncols
    fig_h = 4.0 * nrows
    fig, axs = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axs = np.atleast_1d(axs).reshape(nrows, ncols)

    ims = []
    vmin, vmax = -0.5, J - 0.5
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            ax = axs[r, c]
            if idx >= total:
                ax.axis("off")
                continue
            ax.set_aspect("equal")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_xlabel("x")
            if c == 0:
                ax.set_ylabel("y")
            theta = np.linspace(0, 2 * math.pi, 512)
            ax.plot(np.cos(theta), np.sin(theta), linewidth=1.0, color="k", alpha=0.8)

            # Placeholder image; will set actual data later
            blank = np.zeros((int((extent[3] - extent[2]) * 100), int((extent[1] - extent[0]) * 100)), dtype=int)
            im = ax.imshow(blank, origin="lower", extent=extent, interpolation="nearest", vmin=vmin, vmax=vmax)
            ims.append(im)

            if idx == 0:
                ax.set_title("Ground truth")
            else:
                ax.set_title(f"head ∘ layer^{idx-1}(x)")
            idx += 1

    fig.subplots_adjust(right=0.90)
    cb = fig.colorbar(ims[0], ax=axs.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_ticks(range(J))
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 1.0])
    plt.ion(); plt.show(block=False)
    fig.canvas.draw(); fig.canvas.flush_events()
    return fig, axs, ims

def collect_anchors_interactive(J: int, anchors_per_class: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Interactive anchor selection on unit disk.
    Returns anchors_xy: (M, 2) and anchor_classes: (M,)
    """
    total_needed = J * anchors_per_class
    anchors: list[tuple[float, float]] = []
    classes: list[int] = []

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.0))
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    theta = np.linspace(0, 2 * math.pi, 512)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1.0, color="k", alpha=0.8)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top",
                  fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    scatters = []
    for j in range(J):
        sc = ax.scatter([], [], s=50, color=colors[j % len(colors)], alpha=0.9, label=f"class {j}")
        scatters.append(sc)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    current_idx = 0
    def update_text():
        cls = current_idx // anchors_per_class
        left_in_cls = anchors_per_class - (current_idx % anchors_per_class)
        txt.set_text(f"Click {left_in_cls} anchor(s) for class {cls} (of {J}); total {current_idx}/{total_needed}")

    def on_click(event):
        nonlocal current_idx
        if current_idx >= total_needed:
            return
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if x * x + y * y > 1.0:
            return
        anchors.append((x, y))
        cls = current_idx // anchors_per_class
        classes.append(cls)
        sc = scatters[cls]
        off = sc.get_offsets()
        if off is None or len(off) == 0:
            new_off = np.array([[x, y]], dtype=float)
        else:
            new_off = np.vstack([off, np.array([[x, y]], dtype=float)])
        sc.set_offsets(new_off)
        current_idx += 1
        update_text()
        fig.canvas.draw_idle()
        if current_idx >= total_needed:
            plt.close(fig)

    update_text()
    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    try:
        fig.canvas.mpl_disconnect(cid)
    except Exception:
        pass

    anchors_xy = np.array(anchors, dtype=float).reshape(-1, 2)
    anchor_classes = np.array(classes, dtype=int).reshape(-1)
    return anchors_xy, anchor_classes

def assign_labels_by_anchors_xy(XY: np.ndarray, anchors_xy: np.ndarray, anchor_classes: np.ndarray) -> np.ndarray:
    diffs = XY[:, None, :] - anchors_xy[None, :, :]
    d2 = np.sum(diffs * diffs, axis=-1)
    nn = np.argmin(d2, axis=1)
    return anchor_classes[nn]

def anchors_xy_to_xyz(anchors_xy: np.ndarray) -> np.ndarray:
    """
    Map 2D disk anchors to hemisphere points by z = sqrt(1 - x^2 - y^2).
    Returns (M, 3).
    """
    x = anchors_xy[:, 0]
    y = anchors_xy[:, 1]
    r2 = x * x + y * y
    z = np.sqrt(np.maximum(1.0 - r2, 0.0))
    return np.stack([x, y, z], axis=-1)

def assign_labels_by_anchors_xyz(P: np.ndarray, anchors_xyz: np.ndarray, anchor_classes: np.ndarray) -> np.ndarray:
    """
    P: (N, 3) points on hemisphere (unit or near-unit)
    anchors_xyz: (M, 3) anchor positions on hemisphere
    Returns (N,) nearest-anchor classes by Euclidean distance in R^3.
    """
    diffs = P[:, None, :] - anchors_xyz[None, :, :]  # (N, M, 3)
    d2 = np.sum(diffs * diffs, axis=-1)  # (N, M)
    nn = np.argmin(d2, axis=1)
    return anchor_classes[nn]


@torch.no_grad()
def compute_grid_labels_model(model: SparseStack, pts: np.ndarray, device: torch.device) -> np.ndarray:
    # pts: (D, N) numpy
    x = torch.from_numpy(pts.T).to(device=device, dtype=torch.float32)  # (N, D)
    x = l2_normalize(x, dim=-1)
    logits, _ = model(x)
    pred = torch.argmax(logits, dim=-1).detach().cpu().numpy()  # (N,)
    return pred

@torch.no_grad()
def compute_grid_labels_model_views(model: SparseStack, pts: np.ndarray, device: torch.device) -> list[np.ndarray]:
    """
    Returns list of length (L+1):
      labels[0] = argmax(head(x))
      labels[t] = argmax(head(layer_{t-1}(...layer_0(x)))) for t >= 1
    """
    x = torch.from_numpy(pts.T).to(device=device, dtype=torch.float32)  # (N, D)
    x = l2_normalize(x, dim=-1)
    views: list[np.ndarray] = []
    h = x
    # t = 0
    logits0 = model.head(h)
    views.append(torch.argmax(logits0, dim=-1).detach().cpu().numpy())
    # t = 1..L
    for layer in model.layers:
        h, _ = layer(h)
        logits = model.head(h)
        views.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())
    return views


@torch.no_grad()
def compute_grid_labels_gt(gt_model: GroundTruthClassifier, pts: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(pts.T).to(device=device, dtype=torch.float32)
    x = l2_normalize(x, dim=-1)
    logits = gt_model(x)
    labels = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    return labels


# ----------------------------
# Training loop with live visualization
# ----------------------------
def run_training_viz(
    D: int = 3,
    J: int = 6,
    L: int = 2,
    m: int = 32,
    b: int = 1,
    k: int = 4,
    lambda_coeff: float = 1.0,
    selection_relu: bool = False,
    aux_coeff: float = 0.0,
    use_anchor_gt: bool = True,
    anchors_per_class: int = 3,
    plot_all_layer_views: bool = False,
    grid_res: int = 321,
    steps: int = 4000,
    batch_size: int = 256,
    lr: float = 3e-3,
    viz_every: int = 20,
    seed: int = 123,
    device: torch.device | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build teacher to produce ground-truth labels
    anchors_xy_np: np.ndarray | None = None
    anchor_classes_np: np.ndarray | None = None
    anchors_xyz_np: np.ndarray | None = None
    if use_anchor_gt:
        print(f"Anchor-based GT: Click {anchors_per_class} anchors for each of {J} classes.")
        anchors_xy_np, anchor_classes_np = collect_anchors_interactive(J=J, anchors_per_class=anchors_per_class)
        anchors_xyz_np = anchors_xy_to_xyz(anchors_xy_np)  # (M, 3)
    else:
        raise ValueError()
        # gt = GroundTruthClassifier(D=D, J=J, hidden_dim=gt_hidden_dim, num_hidden_layers=gt_num_hidden_layers, seed=seed).to(device)
        # gt.eval()

    # Build student model
    model = SparseStack(D=D, J=J, L=L, m=m, b=b, k=k, lambda_coeff=lambda_coeff, selection_relu=selection_relu).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Prepare grid and ground-truth labels image
    X, Y, mask = unit_disk_grid(grid_res)
    pts0 = lift_to_upper_hemisphere(X, Y, mask)  # (3, N)
    # Figures
    unified_mode = bool(plot_all_layer_views)
    if unified_mode:
        fig_all, axs_all, ims_all = make_unified_views_figure(J=J, num_layers=int(model.L), extent=(-1, 1, -1, 1))
        fig = fig_all
        axs = axs_all
        im_gt = ims_all[0]
        ims_views = ims_all[1:]  # length L+1
    else:
        fig, axs, im_gt, im_model = make_boundary_figures(X, mask, J=J)
        ims_views = None

    # Compute and render ground-truth labels once
    if use_anchor_gt:
        # Use 3D hemisphere points for labeling
        P_flat = pts0.T  # (N, 3)
        gt_labels_flat = assign_labels_by_anchors_xyz(P_flat, anchors_xyz_np, anchor_classes_np)
        G_gt = labels_to_grid(gt_labels_flat, X, mask)
    else:
        gt_labels = compute_grid_labels_gt(gt, pts0, device=device)  # (N,)
        G_gt = labels_to_grid(gt_labels, X, mask)
    im_gt.set_data(np.ma.array(G_gt, mask=(G_gt < 0)))
    if use_anchor_gt:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # overlay anchors on the first axis (GT panel)
        gt_ax = axs[0, 0] if unified_mode else axs[0]
        for j in range(J):
            pts_j = anchors_xy_np[anchor_classes_np == j]
            if pts_j.size == 0:
                continue
            gt_ax.scatter(pts_j[:, 0], pts_j[:, 1], s=55, color=colors[j % len(colors)], alpha=0.95, edgecolors="k", linewidths=0.5)
    fig.canvas.draw(); fig.canvas.flush_events()

    # Training data sampler
    def batch():
        x = sample_upper_hemisphere(batch_size, D, device=device)  # (B, D)
        if use_anchor_gt:
            # Use 3D nearest neighbor on hemisphere
            x3 = x[:, :3]  # (B, 3)
            anchors_xyz = torch.tensor(anchors_xyz_np, device=x.device, dtype=x.dtype)  # (M, 3)
            diffs = x3.unsqueeze(1) - anchors_xyz.unsqueeze(0)  # (B, M, 3)
            d2 = (diffs * diffs).sum(dim=-1)  # (B, M)
            idx = torch.argmin(d2, dim=-1)
            anchor_classes = torch.tensor(anchor_classes_np, device=x.device, dtype=torch.long)
            y = anchor_classes.index_select(0, idx)
        else:
            with torch.no_grad():
                y_logits = gt(x)
                y = torch.argmax(y_logits, dim=-1)
        return x, y

    # Train
    model.train()
    for t in range(1, steps + 1):
        xb, yb = batch()
        logits, aux = model(xb)
        task = F.cross_entropy(logits, yb)
        # batch accuracy
        preds = torch.argmax(logits, dim=-1)
        batch_acc = (preds == yb).to(torch.float32).mean()

        # aux losses
        aux_loss = torch.zeros((), device=logits.device)
        for kname, val in aux.items():
            if kname.endswith(AUX_LOSS_SUFFIX):
                aux_loss = aux_loss + val
        # normalize aux by number of layers to keep scale consistent
        num_layers = getattr(model, "L", 1)
        if num_layers > 0:
            aux_loss = aux_loss / float(num_layers)
        loss = task + aux_coeff * aux_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if t % viz_every == 0 or t == 1:
            model.eval()
            if unified_mode:
                labels_list = compute_grid_labels_model_views(model, pts0, device=device)
                for i, lab in enumerate(labels_list):
                    Gv = labels_to_grid(lab, X, mask)
                    ims_views[i].set_data(np.ma.array(Gv, mask=(Gv < 0)))
                fig.canvas.draw(); fig.canvas.flush_events()
                plt.pause(0.001)
            else:
                pred_labels = compute_grid_labels_model(model, pts0, device=device)  # (N,)
                G_model = labels_to_grid(pred_labels, X, mask)
                im_model.set_data(np.ma.array(G_model, mask=(G_model < 0)))
                fig.canvas.draw(); fig.canvas.flush_events()
                plt.pause(0.01)
            model.train()

        if t % 200 == 0:
            aux_raw = float(aux_loss)
            aux_weighted = float(aux_coeff * aux_loss)
            total_loss = float(task + aux_coeff * aux_loss)
            acc_val = float(batch_acc)
            print(f"step {t}: task={float(task):.4f} acc={acc_val:.4f} aux_weighted={aux_weighted:.4f} total_loss={total_loss:.4f}")

    # Final render hold
    plt.ioff()
    plt.show()


# ----------------------------
# Hyperparameters
# ----------------------------
D: int = 3
J: int = 6  # num classes
L: int = 2
m: int = 2000
b: int = 1
k: int = 2
lambda_coeff: float = 1.
# Use ReLU-based selection energy for top-k
selection_relu: bool = True
# gt_hidden_dim: int = 3
# gt_num_hidden_layers: int = 2
aux_coeff: float = 0.02
use_anchor_gt: bool = True
anchors_per_class: int = 6
plot_all_layer_views: bool = True
grid_res: int = 321
steps: int = 60000
batch_size: int = 512
lr: float = 1e-3
viz_every: int = 100
seed: int = 1111

if __name__ == "__main__":
    run_training_viz(
        D=D,
        J=J,
        L=L,
        m=m,
        b=b,
        k=k,
        lambda_coeff=lambda_coeff,
        selection_relu=selection_relu,
        aux_coeff=aux_coeff,
        use_anchor_gt=use_anchor_gt,
        anchors_per_class=anchors_per_class,
        plot_all_layer_views=plot_all_layer_views,
        grid_res=grid_res,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        viz_every=viz_every,
        seed=seed,
    )

