import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Constants
# ----------------------------
GLOBAL_D = 3
GLOBAL_M = 3
GLOBAL_K = 2
NUM_PARTICLES = 128
SEED = 655
BETA = 1. # (discretized version)
grid_res = 501  # increase for sharper boundaries


# ----------------------------
# Dictionary construction
# ----------------------------
def random_rotation(rng: np.random.Generator, D: int = 3) -> np.ndarray:
    """Random DxD rotation matrix (det = +1)."""
    A = rng.standard_normal((D, D))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def tetrahedron_cols() -> np.ndarray:
    """
    3x4 matrix whose columns are unit vectors with pairwise dot = -1/3.
    Regular tetrahedron centered at origin.
    """
    A = np.array(
        [
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ],
        dtype=float,
    )
    A /= np.linalg.norm(A[:, 0])
    return A  # 3x4


def make_low_coherence_dicts(rng: np.random.Generator, D: int, m: int):
    """
    Returns V, U, W in R^{D x m} with low pairwise coherence.
    For D=3, m=4, tetrahedron is the canonical low-coherence choice.
    """
    # create Dxm base matrix with low coherence
    if m == 4 and D == 3:
        base = tetrahedron_cols()
    elif m == D:
        base = np.eye(D)
    else:
        # just use a random matrix with columns of unit norm
        base = rng.standard_normal((D, m))
        base = base / np.linalg.norm(base, axis=0, keepdims=True)
    
    V = random_rotation(rng, D) @ base  # D x m
    # U = random_rotation(rng, D) @ base
    U = build_U_no_fixed_points(rng, V)
    print("Successfully built U and V")
    W = random_rotation(rng, D) @ base
    return V, U, W


def pairwise_dot_stats(A: np.ndarray, name: str) -> None:
    G = A.T @ A
    off = G - np.eye(G.shape[0])
    max_abs = np.max(np.abs(off))
    print(f"{name}: max |col_i^T col_j| off-diag = {max_abs:.4f}")
    print(f"{name} Gram off-diag:\n{np.round(off, 4)}\n")


def build_U_no_fixed_points(
    rng: np.random.Generator,
    V: np.ndarray,              # (D, m)
    max_tries: int = 2000,
    avoid_align: float = 0.999, # avoid u_i ~ v_i (makes a_i tiny)
):
    D, m = V.shape
    U = np.zeros((D, m), dtype=float)

    for i in range(m):
        v_i = V[:, i]

        for _ in range(max_tries):
            u = rng.standard_normal(D)
            u /= np.linalg.norm(u) + 1e-12

            # avoid u too aligned with v_i (optional but helps numerically)
            if abs(float(u @ v_i)) > avoid_align:
                continue

            a = u - v_i
            na = np.linalg.norm(a)
            if na < 1e-8:
                continue
            a_hat = a / na

            # fixed-point candidate occurs when a_hat routes to i
            # check if score of i is within tol of largest
            winner_idx, scores = winners_abs(V, a_hat.reshape(-1, 1))  # (1,), (m, 1)
            widx = int(winner_idx[0])
            if np.abs(scores[i, 0]) >= np.abs(scores[widx, 0]) - 1e-2:
                continue

            U[:, i] = u
            break
        else:
            raise RuntimeError(f"Failed to sample u_{i} satisfying constraints; relax thresholds.")

    return U

# ----------------------------
# Geometry / grid utilities
# ----------------------------
def unit_disk_grid(res: int):
    """Return X, Y meshgrid, and boolean mask for the unit disk."""
    xs = np.linspace(-1.0, 1.0, res)
    ys = np.linspace(-1.0, 1.0, res)
    X, Y = np.meshgrid(xs, ys)
    mask = (X**2 + Y**2) <= 1.0
    return X, Y, mask


def lift_to_upper_hemisphere(X: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Map points (x,y) in unit disk to (x,y,z) on upper hemisphere with z = sqrt(1-x^2-y^2).
    Returns pts with shape (3, N).
    """
    R2 = X**2 + Y**2
    Z = np.zeros_like(X)
    Z[mask] = np.sqrt(1.0 - R2[mask])
    pts = np.stack([X[mask], Y[mask], Z[mask]], axis=0)  # 3 x N
    return pts


# ----------------------------
# Layer ops
# ----------------------------
def winners_abs(V: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Winner-take-all by max |v_i^T x|.
    V is (D, m) and pts is (D, N).
    Returns (winner_indices (N,), raw_scores (m, N)) where raw_scores are signed v_i^T x.
    """
    scores = V.T @ pts  # (m, N)
    winner_idxs = np.argmax(np.abs(scores), axis=0)  # (N,)
    return winner_idxs, scores


def apply_replacement(V: np.ndarray, U: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Top-k replacement operator (matching pursuit style):
      For t = 1..k:
        - pick i_t = argmax |v_i^T r_{t-1}| where r_0 = x
        - c_t = v_{i_t}^T r_{t-1} (signed)
        - r_t = r_{t-1} - c_t v_{i_t}  (deflation step)
      Finally:
        x' = normalize_cols(r_k + sum_t c_t u_{i_t})
      When k=1, this reduces to: x' = x - c v_i + c u_i
    """
    # number of atoms to select
    k = int(min(GLOBAL_K, V.shape[1]))
    if k <= 0:
        return pts

    D, m = V.shape
    N = pts.shape[1]

    # Residual starts as the input
    residual = pts.copy()  # (D, N)

    # Store selections and coefficients per step
    sel_idxs = np.empty((k, N), dtype=int)
    sel_coeffs = np.zeros((k, N), dtype=pts.dtype)

    # Greedy selection with deflation (matching pursuit)
    for t in range(k):
        scores = V.T @ residual  # (m, N)
        winners = np.argmax(np.abs(scores), axis=0)  # (N,)
        c = scores[winners, np.arange(N)]  # (N,)

        sel_idxs[t] = winners
        sel_coeffs[t] = c

        # Deflate residual by subtracting the selected v scaled by its coefficient
        vsel = V[:, winners]  # (D, N)
        residual = residual - vsel * c  # broadcast over rows

    # Start from r_k (x minus sum c_t v_{i_t})
    pts_prime = residual

    # Add sum_t c_t u_{i_t}
    for t in range(k):
        winners = sel_idxs[t]
        c = sel_coeffs[t]
        usel = U[:, winners]  # (D, N)
        pts_prime = pts_prime + usel * c
    
    # Ensure unit-norm outputs per column
    pts_prime = normalize_cols(pts_prime)
    return pts_prime


def iterate_layer(V: np.ndarray, U: np.ndarray, pts: np.ndarray, T: int) -> list[np.ndarray]:
    """
    Returns list of iterates: [x0, x1, ..., xT]
    where x_{t+1} = L(x_t) and L is apply_replacement(V, U, ·).
    """
    xs = [pts]
    cur = pts
    for _ in range(T):
        cur = apply_replacement(V, U, cur)
        xs.append(cur)
    return xs


# ----------------------------
# Plotting
# ----------------------------
def labels_to_grid(
    labels: np.ndarray, X: np.ndarray, mask: np.ndarray, fill_value: int = -1
) -> np.ndarray:
    """Place length-N labels back into an (H,W) grid with mask."""
    out = np.full(X.shape, fill_value=fill_value, dtype=int)
    out[mask] = labels
    return out


def draw_unit_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1)


def plot_label_grids(
    grids: list[np.ndarray],
    titles: list[str],
    m: int,
    extent=(-1, 1, -1, 1),
    max_cols: int = 4,
    figsize_per_panel=(4.2, 3.8),
    title_fontsize: int = 10,
):
    """
    Plot multiple categorical label grids in a tiled layout with a shared colorbar.
    grids: list of (H,W) int arrays with -1 outside disk.
    """
    assert len(grids) == len(titles)
    n = len(grids)

    # Choose a nice tiling automatically.
    ncols = min(max_cols, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    # Use consistent discrete-ish scaling for labels 0..m-1
    vmin, vmax = -0.5, m - 0.5

    ims = []
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        if idx >= n:
            ax.axis("off")
            continue

        G = np.ma.array(grids[idx], mask=(grids[idx] < 0))
        im = ax.imshow(
            G,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)

        ax.set_title(titles[idx], fontsize=title_fontsize)
        ax.set_aspect("equal")

        # Reduce clutter: only label outer axes
        if r == nrows - 1:
            ax.set_xlabel("x")
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])

        draw_unit_circle(ax)

    # Shared colorbar for all axes (much cleaner for many panels)
    cb = fig.colorbar(ims[0], ax=axes, fraction=0.025, pad=0.02)
    cb.set_ticks(range(m))
    cb.set_label("winner index")

    fig.tight_layout()
    plt.show()

# Dynamical particle simulation
# ----------------------------
def normalize_cols(pts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize columns of a (D, N) array to unit norm."""
    norms = np.linalg.norm(pts, axis=0, keepdims=True)
    return pts / np.maximum(norms, eps)


def animate_particles_unit_disk(
    V: np.ndarray,         # (D, m)
    U: np.ndarray,         # (D, m)
    X0: np.ndarray,        # (D, num_particles) unit vectors
    steps: int = 50000,
    stride: int = 5,       # update plot every stride steps
    tail: int|None = None,        # show last `tail` steps of each trajectory
    anchor_size: float = 55.0,
    particle_size: float = 28.0,
    pause_time: float = 0.005,
):
    """
    Live animation of particle trajectories under:
      x <- normalize(x + beta L(x))
    Plots (x1,x2) on the unit disk.
    """
    if tail is None:
        tail = steps

    X = normalize_cols(X0.copy())
    D, K = X.shape

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    draw_unit_circle(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # Anchor directions (project to (x1,x2))
    P = V[:2, :]  # 2 x m
    z = V[2, :]
    mask_pos = z >= 0
    ax.scatter(P[0, mask_pos], P[1, mask_pos], marker="+", linewidths=1.8, alpha=0.85)
    ax.scatter(P[0, ~mask_pos], P[1, ~mask_pos], marker="x", linewidths=1.8, alpha=0.85)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Initialize artists
    lines = []
    heads = []
    history = [np.empty((0, 2)) for _ in range(K)]

    for j in range(K):
        c = colors[j % len(colors)]
        (ln,) = ax.plot([], [], linewidth=1.0, color=c, alpha=0.35)  # tail line
        sc = ax.scatter([], [], s=particle_size, color=c, marker="o")  # current position
        lines.append(ln)
        heads.append(sc)

    plt.ion()
    plt.show()

    # pause for 2 seconds before starting the animation
    plt.pause(2.0)

    for t in range(steps):
        # one dynamics step
        LX = apply_replacement(V, U, X)
        X = normalize_cols(X + BETA * LX)

        if t % stride != 0:
            continue

        # update history and artists
        for j in range(K):
            xy = X[:2, j]
            history[j] = np.vstack([history[j], xy[None, :]])
            if history[j].shape[0] > tail:
                history[j] = history[j][-tail:, :]

            lines[j].set_data(history[j][:, 0], history[j][:, 1])
            heads[j].set_offsets(xy[None, :])

        fig.canvas.draw_idle()
        plt.pause(pause_time)  # controls animation speed / UI responsiveness

    plt.ioff()
    plt.show()

def interactive_particle_viewer(
    V: np.ndarray,         # (D, m)
    U: np.ndarray,         # (D, m)
    tail: int | None = None,
    particle_size: float = 40.0,
    pause_time: float = 0.01,   # seconds per timer tick
):
    """
    Interactive viewer: click inside the unit disk to launch a particle and
    watch its trajectory evolve under x <- normalize(x + BETA * L(x)).
    Each click starts a NEW trajectory that persists; previous trajectories remain visible.
    """
    D, m = V.shape
    if D < 3:
        raise ValueError("interactive_particle_viewer expects D >= 3 for unit-disk -> hemisphere lift.")

    # Figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    draw_unit_circle(ax)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Click inside the unit disk to launch a particle")

    # Anchor directions (project to (x1,x2))
    P = V[:2, :]  # 2 x m
    z = V[2, :]
    mask_pos = z >= 0
    ax.scatter(P[0, mask_pos], P[1, mask_pos], marker="+", linewidths=1.6, alpha=0.8)
    ax.scatter(P[0, ~mask_pos], P[1, ~mask_pos], marker="x", linewidths=1.6, alpha=0.8)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # State for all particles (persist across clicks)
    particles: list[dict] = []
    color_idx = 0

    def reset_particle(xy: tuple[float, float]):
        nonlocal particles, color_idx
        x, y = xy
        r2 = x * x + y * y
        if r2 > 1.0:
            return
        # Lift to upper hemisphere (z >= 0), pad to D dims
        z_val = np.sqrt(max(1.0 - r2, 0.0))
        X = np.zeros((D, 1), dtype=float)
        X[0, 0] = x
        X[1, 0] = y
        X[2, 0] = z_val
        X = normalize_cols(X)
        history = np.empty((0, 2))

        # Create a new trajectory with its own color and artists
        c = colors[color_idx % len(colors)]
        color_idx += 1
        (line,) = ax.plot([], [], linewidth=1.0, color=c, alpha=0.35)
        head = ax.scatter([], [], s=particle_size, color=c, marker="o")
        head.set_offsets(np.array([[x, y]]))

        particles.append({"X": X, "history": history, "line": line, "head": head})
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        reset_particle((float(event.xdata), float(event.ydata)))

    cid = fig.canvas.mpl_connect("button_press_event", on_click)

    def step():
        # Update all active particles
        if not particles:
            return
        for p in particles:
            X = p["X"]
            history = p["history"]
            line = p["line"]
            head = p["head"]
            # One dynamics step
            LX = apply_replacement(V, U, X)  # (D, 1)
            X = normalize_cols(X + BETA * LX)
            p["X"] = X
            xy = X[:2, 0]
            history = np.vstack([history, xy[None, :]])
            if tail is not None and history.shape[0] > tail:
                history = history[-tail:, :]
            p["history"] = history
            # Update artists
            line.set_data(history[:, 0], history[:, 1])
            head.set_offsets(xy[None, :])
        fig.canvas.draw_idle()

    # Use a timer to update the trajectory continuously
    timer = fig.canvas.new_timer(interval=int(max(1, pause_time * 1000)))
    timer.add_callback(step)
    timer.start()

    try:
        plt.show()
    finally:
        # Clean up connections and timer when the window closes
        try:
            fig.canvas.mpl_disconnect(cid)
        except Exception:
            pass
        try:
            timer.stop()
        except Exception:
            pass


# ----------------------------
# Boundaries
# ----------------------------
def plot_boundaries(V: np.ndarray, U: np.ndarray, W: np.ndarray, num_iters: int = 16):
    # V is (D, m)
    X, Y, mask = unit_disk_grid(grid_res)
    pts0 = lift_to_upper_hemisphere(X, Y, mask)  # (3, N)

    # Layer 1 winners on x
    i1, _ = winners_abs(V, pts0)
    G1 = labels_to_grid(i1, X, mask)

    # Apply replacement, then layer 2 winners on x'
    pts1 = apply_replacement(V, U, pts0)
    i2, _ = winners_abs(W, pts1)
    G2 = labels_to_grid(i2, X, mask)

    plot_label_grids(
        grids=[G1, G2],
        titles=[
            "Layer 1 winner: argmax |v_i^T x|",
            "Layer 2 winner: argmax |w_i^T (L(x))|",
        ],
        m=GLOBAL_M,
        max_cols=2,
    )

    # Partition boundaries
    xs = iterate_layer(V, U, pts0, num_iters)
    grids = []
    titles = []
    for t, pts_t in enumerate(xs):
        it, _ = winners_abs(V, pts_t)
        grids.append(labels_to_grid(it, X, mask))
        titles.append(f"Winner under V at iterate t={t}")
    plot_label_grids(grids=grids, titles=titles, m=GLOBAL_M, max_cols=num_iters)
    return 


def simulate_particles(V: np.ndarray, U: np.ndarray, num_particles: int):
    rng = np.random.default_rng(SEED)
    D = V.shape[0]
    # Sample (x, y) uniformly over the unit disk and lift to the unit sphere
    # r ~ sqrt(U[0,1]) ensures area-uniform sampling in the disk
    r = np.sqrt(rng.random(num_particles))
    theta = rng.uniform(0.0, 2.0 * np.pi, size=num_particles)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Choose random sign for z so that (x, y) stays uniform in the disk
    z_sign = rng.choice(np.array([-1.0, 1.0]), size=num_particles)
    z = z_sign * np.sqrt(np.maximum(1.0 - r**2, 0.0))
    X0 = np.zeros((D, num_particles), dtype=float)  # D x N
    X0[0, :] = x
    X0[1, :] = y
    X0[2, :] = z
    X0 = normalize_cols(X0)  # D x N
    animate_particles_unit_disk(V=V, U=U, X0=X0, steps=50000, tail=None)

def animate_decision_boundary_evolution(
    V: np.ndarray,
    U: np.ndarray,
    Xgrid: np.ndarray,          # meshgrid X from unit_disk_grid
    mask: np.ndarray,           # disk mask
    pts0: np.ndarray,           # (3, N) lifted hemisphere points, from lift_to_upper_hemisphere
    steps: int = 2000,
    stride: int = 5,            # update plot every `stride` steps
    figsize=(6.5, 6.0),
    show_colorbar: bool = True,
):
    """
    Animate: pts <- normalize(pts + beta * L(pts))  (vectorized over all grid points)
    and color each original (x,y) by winner under V at the current iterate.

    Notes:
    - This is a *pullback* visualization: the grid stays fixed, colors update based on the
      current iterate of each starting point.
    - For speed, increase `stride` and/or decrease `grid_res`.
    """
    m = V.shape[1]
    pts = pts0.copy()

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    draw_unit_circle(ax)

    # Initial labels
    i0, _ = winners_abs(V, pts)
    G0 = labels_to_grid(i0, Xgrid, mask)
    G0m = np.ma.array(G0, mask=(G0 < 0))

    im = ax.imshow(
        G0m,
        origin="lower",
        extent=[-1, 1, -1, 1],
        interpolation="nearest",
        vmin=-0.5,
        vmax=m - 0.5,
    )

    cb = None
    if show_colorbar:
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_ticks(range(m))

    # Minimal time indicator
    txt = ax.text(
        0.02, 0.98, "t=0",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
    )

    plt.ion()
    plt.show()

    for t in range(1, steps + 1):
        # Evolve all grid points one step:
        # update rule requested: pts <- normalize(pts + beta * L(pts))
        Lpts = apply_replacement(V, U, pts)
        pts = normalize_cols(pts + BETA * Lpts)

        if t % stride != 0:
            continue

        # Recompute winners and update image
        it, _ = winners_abs(V, pts)
        Gt = labels_to_grid(it, Xgrid, mask)
        im.set_data(np.ma.array(Gt, mask=(Gt < 0)))
        txt.set_text(f"t={t}")

        fig.canvas.draw_idle()
        plt.pause(0.001)

    plt.ioff()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    V, U, W = make_low_coherence_dicts(rng, D=GLOBAL_D, m=GLOBAL_M)

    # V is (D, m)

    # experiment 1: plot boundaries
    plot_boundaries(V=V, U=U, W=W, num_iters=16)
    
    # experiment 2: simulate particles
    # simulate_particles(V=V, U=U, num_particles=NUM_PARTICLES)

    # experiment 3 (interactive): click inside the unit disk to launch a particle
    interactive_particle_viewer(V=V, U=U, tail=None, pause_time=0.001)

    X, Y, mask = unit_disk_grid(grid_res)
    pts0 = lift_to_upper_hemisphere(X, Y, mask)
    # animate_decision_boundary_evolution(
    #     V=V,
    #     U=U,
    #     Xgrid=X,
    #     mask=mask,
    #     pts0=pts0,
    #     steps=8000,
    #     stride=25,   # bump this up if it’s slow
    # )

