"""
Interactive neural network activation visualization.
Displays the network structure with nodes colored by activation levels.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button

VOWEL_NAMES = ["ae", "ah", "aw", "eh", "er", "ey", "ih", "iy", "oa", "oo", "uh", "uw"]


def _draw_network_figure(ax, x, h1, h2, probs, true_label, pred_label, sample_idx, total_samples, anim_phase=None, pulse_neurons=True):
    """Draw one static snapshot of the network for given activations.
    anim_phase: 0=highlight Input->H1, 1=highlight H1->H2, 2=highlight H2->Output, None=normal view.
    pulse_neurons: when True and anim_phase set, show glow on destination layer neurons."""
    ax.clear()
    ax.set_facecolor("#0a0a0a")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Dark gray to lime green colormap for activations
    colors_green = ["#1a1a1a", "#0d3d0d", "#1a5c1a", "#2d8a2d", "#4ade4a", "#90ee90"]
    cmap_activation = LinearSegmentedColormap.from_list("activation", colors_green, N=256)
    cmap_input = LinearSegmentedColormap.from_list("input", ["#1a1a1a", "#2d8a2d"], N=256)

    # Store node positions for drawing connections (center x, center y)
    input_pos = []
    h1_pos = []
    h2_pos = []
    output_pos = []

    # Precompute positions first (needed for connection lines)
    for i in range(31):
        input_pos.append((0.5, 1.5 + i * 0.22 + 0.09))
    for j in range(20):
        h1_pos.append((2.5, 1.5 + j * 0.38))
    for j in range(16):
        h2_pos.append((4.5, 2.2 + j * 0.38))
    for i in range(12):
        output_pos.append((6.8, 1.8 + i * 0.6))

    # ---- Connection lines (LineCollection for fast batch drawing) ----
    dim_color, dim_border = "#6a8aaa", "#2a3a4a"
    dim_alpha = 0.15 if anim_phase is not None else 0.2
    highlight_color, highlight_border = "#00e5ff", "#006080"
    highlight_alpha = 0.5

    def make_segments(positions_a, positions_b):
        segs = []
        for (x1, y1) in positions_a:
            for (x2, y2) in positions_b:
                segs.append([[x1, y1], [x2, y2]])
        return np.array(segs)

    def add_line_layer(segs, color, alpha, lw, z=0):
        lc = LineCollection(segs, colors=color, alpha=alpha, linewidths=lw, zorder=z, capstyle="round")
        ax.add_collection(lc)

    segs_ih1 = make_segments(input_pos, h1_pos)
    segs_h1h2 = make_segments(h1_pos, h2_pos)
    segs_h2o = make_segments(h2_pos, output_pos)

    for segs, is_hl in [(segs_ih1, anim_phase == 0), (segs_h1h2, anim_phase == 1), (segs_h2o, anim_phase == 2)]:
        lw = 0.7 if is_hl else 0.5
        a = highlight_alpha if is_hl else dim_alpha
        add_line_layer(segs, highlight_border if is_hl else dim_border, a * 0.9, lw + 0.8)
        add_line_layer(segs, highlight_color if is_hl else dim_color, a, lw)

    # ---- Input strip (31 features) ----
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    for i in range(31):
        val = float(x_norm[i])
        color = cmap_input(val)
        y_rect = 1.5 + i * 0.22
        rect = Rectangle((0.3, y_rect), 0.4, 0.18, facecolor=color, edgecolor="#2a4a6a")
        ax.add_patch(rect)
    ax.text(0.15, 8.9, "31", fontsize=10, color="#7aa", va="center")
    ax.text(0.85, 5, "Input", fontsize=9, color="#8ab", va="center", ha="center", rotation=90)

    # ---- Hidden layer 1 (subsample to 20 nodes) ----
    n_h1 = len(h1)
    idx_h1 = np.linspace(0, n_h1 - 1, 20, dtype=int)
    h1_max = max(h1.max(), 1e-6)
    pulse_layer = anim_phase == 0 and pulse_neurons
    h1_winner_j = int(np.argmax([h1[i] for i in idx_h1])) if pulse_layer else -1
    for j, i in enumerate(idx_h1):
        val = float(h1[i] / h1_max)
        color = cmap_activation(val)
        y_pos = 1.5 + j * 0.38
        is_highlighted = pulse_layer and j == h1_winner_j
        if is_highlighted:
            glow = Circle((2.5, y_pos), 0.22, facecolor="none", edgecolor="#00d4ff", linewidth=2, alpha=0.7, zorder=0.5)
            ax.add_patch(glow)
        circle = Circle((2.5, y_pos), 0.12, facecolor=color, edgecolor="#00d4ff" if is_highlighted else "#2a4a6a", linewidth=1.5 if is_highlighted else 1)
        ax.add_patch(circle)
    ax.text(2.5, 8.9, "128", fontsize=10, color="#7aa", ha="center", va="center")
    ax.text(2.5, 0.5, "H1", fontsize=9, color="#8ab", ha="center")

    # ---- Hidden layer 2 (subsample to 16 nodes) ----
    n_h2 = len(h2)
    idx_h2 = np.linspace(0, n_h2 - 1, 16, dtype=int)
    h2_max = max(h2.max(), 1e-6)
    pulse_layer = anim_phase == 1 and pulse_neurons
    h2_winner_j = int(np.argmax([h2[i] for i in idx_h2])) if pulse_layer else -1
    for j, i in enumerate(idx_h2):
        val = float(h2[i] / h2_max)
        color = cmap_activation(val)
        y_pos = 2.2 + j * 0.38
        is_highlighted = pulse_layer and j == h2_winner_j
        if is_highlighted:
            glow = Circle((4.5, y_pos), 0.22, facecolor="none", edgecolor="#00d4ff", linewidth=2, alpha=0.7, zorder=0.5)
            ax.add_patch(glow)
        circle = Circle((4.5, y_pos), 0.12, facecolor=color, edgecolor="#00d4ff" if is_highlighted else "#2a4a6a", linewidth=1.5 if is_highlighted else 1)
        ax.add_patch(circle)
    ax.text(4.5, 8.9, "64", fontsize=10, color="#7aa", ha="center", va="center")
    ax.text(4.5, 1.5, "H2", fontsize=9, color="#8ab", ha="center")

    # ---- Output layer (12 nodes) ----
    pulse_layer = anim_phase == 2 and pulse_neurons
    out_winner = pred_label  # predicted class is the most activated output
    for i in range(12):
        val = float(probs[i])
        color = cmap_activation(val)
        y_pos = 1.8 + i * 0.6
        is_highlighted = pulse_layer and i == out_winner
        if is_highlighted:
            glow = Circle((6.8, y_pos), 0.26, facecolor="none", edgecolor="#00d4ff", linewidth=2, alpha=0.7, zorder=0.5)
            ax.add_patch(glow)
        circle = Circle((6.8, y_pos), 0.14, facecolor=color, edgecolor="#00d4ff" if is_highlighted else "#2a4a6a", linewidth=1.5 if is_highlighted else 1)
        ax.add_patch(circle)
        ax.text(6.8, y_pos, VOWEL_NAMES[i], fontsize=8, color="#fff" if val > 0.3 else "#8ab",
                ha="center", va="center")
        if i == pred_label:
            ax.plot(7.1, y_pos, "o", markersize=6, color="#ffcc00", markeredgecolor="#ffa")
    ax.text(6.8, 8.9, "12", fontsize=10, color="#7aa", ha="center", va="center")
    ax.text(6.8, 0.8, "Output", fontsize=9, color="#8ab", ha="center")

    # ---- Header (placed above layer labels to avoid overlap) ----
    correct = "Correct" if true_label == pred_label else "Incorrect"
    color_status = "#4a4" if true_label == pred_label else "#c44"
    ax.text(5, 9.75, f"Sample {sample_idx + 1} / {total_samples}", fontsize=11, color="#ccc", ha="center", va="center")
    ax.text(5, 9.35, f"True: {VOWEL_NAMES[true_label]}  |  Predicted: {VOWEL_NAMES[pred_label]}  |  {correct}",
            fontsize=10, color=color_status, ha="center", va="center")


def launch_visualizer(model, x, y):
    """Launch interactive visualization. Step through samples with Next/Previous buttons."""
    n_samples = len(y)
    if n_samples == 0:
        print("No samples to visualize.")
        return

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor("#0a0a0a")
    ax_main = fig.add_axes([0.05, 0.15, 0.9, 0.8])
    ax_prev = fig.add_axes([0.35, 0.02, 0.15, 0.05])
    ax_next = fig.add_axes([0.52, 0.02, 0.15, 0.05])

    current_idx = [0]  # use list for closure

    ANIM_DURATION = 0.5

    def update(anim_phase=None, pulse_neurons=True):
        idx = current_idx[0] % n_samples
        xi = x[idx : idx + 1]
        yi = int(y[idx])
        x_vec, h1, h2, probs = model.forward_single(xi.squeeze())
        pred = int(np.argmax(probs))
        _draw_network_figure(ax_main, x_vec, h1, h2, probs, yi, pred, idx, n_samples,
                            anim_phase=anim_phase, pulse_neurons=pulse_neurons)
        fig.canvas.draw_idle()

    def animate_transition():
        """Run propagation animation: highlight connections, then highlight activated neurons for each layer."""
        for phase in [0, 1, 2]:
            update(anim_phase=phase, pulse_neurons=False)  # connections first
            plt.pause(ANIM_DURATION)
            update(anim_phase=phase, pulse_neurons=True)   # then neurons
            plt.pause(ANIM_DURATION)
        update(anim_phase=None)  # final static view

    def on_next(event):
        current_idx[0] += 1
        animate_transition()

    def on_prev(event):
        current_idx[0] -= 1
        animate_transition()

    btn_prev = Button(ax_prev, "Previous", color="#2a4a6a", hovercolor="#3a5a7a")
    btn_next = Button(ax_next, "Next", color="#2a4a6a", hovercolor="#3a5a7a")
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    update()
    plt.show(block=True)
