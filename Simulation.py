"""
Blinking vortex flow (Aref, 1984) with RK4 integration.

Two point vortices at (-a, 0) and (+a, 0) switch on alternately every T/2.
200 passive tracers are advected for 15 s of simulated time.

Usage:
    python blinking_vortex.py            # shows animation, saves GIF + final-frame PNG
    python blinking_vortex.py --no-show  # headless, only saves files
Requires: numpy, matplotlib (pillow for the GIF)
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------- parameters ----------------
N = 200          # tracer particles
T_END = 15.0     # total simulated time [s]
DT = 0.005       # RK4 step [s]
T = 1.2          # blink period [s]
GAMMA = 14.0     # vortex circulation
A = 1.0          # half separation of the vortices
EPS = 0.01       # core softening (avoids the 1/r singularity)
FPS = 30         # animation frame rate (1 simulated second = 1 video second)


def velocity(pos, k):
    """Velocity of all tracers (N,2) induced by vortex k (0 = left, 1 = right)."""
    cx = -A if k == 0 else A
    dx = pos[:, 0] - cx
    dy = pos[:, 1]
    r2 = dx * dx + dy * dy + EPS
    g = GAMMA / (2.0 * np.pi)
    return np.column_stack((-g * dy / r2, g * dx / r2))


def rk4_step(pos, h, k):
    """One classical fourth-order Runge-Kutta step with vortex k active."""
    k1 = velocity(pos, k)
    k2 = velocity(pos + 0.5 * h * k1, k)
    k3 = velocity(pos + 0.5 * h * k2, k)
    k4 = velocity(pos + h * k3, k)
    return pos + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def active_vortex(t):
    return int(np.floor((t + 1e-9) / (T / 2.0))) % 2


def simulate():
    """Integrate all tracers; returns times (M,) and trajectories (M,N,2)."""
    rng = np.random.default_rng(7)
    r = 2.6 * np.sqrt(rng.random(N))
    th = 2 * np.pi * rng.random(N)
    pos = np.column_stack((r * np.cos(th), r * np.sin(th)))
    colour_flag = pos[:, 1] > 0            # colour by initial half-plane

    frame_dt = 1.0 / FPS
    times, traj = [0.0], [pos.copy()]
    t, next_frame = 0.0, frame_dt
    half = T / 2.0
    while t < T_END - 1e-12:
        to_switch = half - (t % half)                  # never step across a blink
        h = min(DT, T_END - t, to_switch, next_frame - t)
        if h <= 1e-12:
            h = min(DT, T_END - t)
        pos = rk4_step(pos, h, active_vortex(t))
        t += h
        if t >= next_frame - 1e-9:
            times.append(t)
            traj.append(pos.copy())
            next_frame += frame_dt
    return np.array(times), np.array(traj), colour_flag


def animate(times, traj, flag, show=True):
    colours = np.where(flag, "#ff7a45", "#2ec4d6")
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    fig.patch.set_facecolor("#0d1b24")
    ax.set_facecolor("#0d1b24")
    L = 5
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    scat = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=14, c=colours, zorder=3)
    trail = 12                                            # frames of trail
    lines = [ax.plot([], [], lw=0.6, alpha=0.35, c=colours[i], zorder=2)[0] for i in range(N)]
    vort = [ax.plot([c], [0], "o", mfc="none", mec="white", ms=12, mew=2, zorder=4)[0]
            for c in (-A, A)]
    title = ax.set_title("", color="white")

    def update(f):
        scat.set_offsets(traj[f])
        lo = max(0, f - trail)
        for i, ln in enumerate(lines):
            ln.set_data(traj[lo:f + 1, i, 0], traj[lo:f + 1, i, 1])
        k = active_vortex(times[f])
        for j, v in enumerate(vort):
            v.set_markerfacecolor("white" if j == k else "none")
        title.set_text(f"t = {times[f]:5.2f} s   active vortex: {'left' if k == 0 else 'right'}")
        return [scat, *lines, *vort, title]

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / FPS, blit=False)
    anim.save("blinking_vortex.gif", writer=PillowWriter(fps=FPS), dpi=80)
    update(len(times) - 1)
    fig.savefig("blinking_vortex_final.png", dpi=150, facecolor=fig.get_facecolor())
    print("Saved blinking_vortex.gif and blinking_vortex_final.png")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    t, X, flag = simulate()
    print(f"Simulated {N} tracers for {t[-1]:.2f} s ({len(t)} frames)")
    animate(t, X, flag, show=not args.no_show)
