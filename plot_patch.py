#!/usr/bin/env python3
"""Gain-plot figures for the FEM patch-array model.

Usage:
  plot_patch.py s11.json            -> S11 sweep figure
  plot_patch.py pattern.json        -> gain heatmap + E/H-plane cuts
Inputs are JSON dumps written by the vitest scratch scripts.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_s11(path: str) -> None:
    with open(path) as f:
        data = json.load(f)
    pts = data["points"]
    f = np.array([p["fMHz"] for p in pts])
    s11 = np.array([p["s11db"] for p in pts if p["s11db"] is not None])
    ff = f[: len(s11)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ff, s11, "o-", lw=1.8)
    ax.axhline(-10, color="gray", ls="--", lw=1)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title(f"S11 of inset-fed patch (L={data['lengthM']*1000:.1f} mm, "
                 f"inset={data['insetM']*1000:.1f} mm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = path.replace(".json", "-s11.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


def plot_pattern(path: str) -> None:
    with open(path) as f:
        data = json.load(f)
    ff = data["farField"]
    theta = np.array(ff["thetaDeg"])
    phi = np.array(ff["phiDeg"])
    gain = np.array(ff["gainDbi"])
    title = data.get("title", "FEM far-field pattern")

    # --- heatmap (theta vs phi) ---
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vmax = np.ceil(np.nanmax(gain))
    im = ax.pcolormesh(phi, theta, gain, shading="auto",
                       vmin=vmax - 40, vmax=vmax, cmap="turbo")
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("theta (deg)")
    ax.set_title(f"{title}\nIEEE gain (dBi), max {ff['maxGainDbi']:.2f} dBi "
                 f"at theta={ff['maxGainThetaDeg']:.1f} deg, phi={ff['maxGainPhiDeg']:.1f} deg")
    fig.colorbar(im, ax=ax, label="dBi")
    fig.tight_layout()
    out = path.replace(".json", "-heatmap.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)

    # --- principal-plane cuts: E-plane (phi=0) and H-plane (phi=90) ---
    def cut(phi_target: float) -> np.ndarray:
        j = int(np.argmin(np.abs(phi - phi_target)))
        return gain[:, j]

    e_plane = cut(0.0)
    h_plane = cut(90.0)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="polar")
    th = np.deg2rad(theta)
    # polar plot needs 0..2pi; mirror the 0..pi cut
    th_full = np.concatenate([th, th[1:-1] + np.pi])
    for arr, label in [(e_plane, "E-plane (phi=0)"), (h_plane, "H-plane (phi=90)")]:
        full = np.concatenate([arr, arr[1:-1]])
        ax.plot(th_full, full, label=label, lw=1.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(vmax - 40, vmax)
    ax.set_title(f"{title}: principal-plane gain cuts (dBi)", pad=18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = path.replace(".json", "-cuts.png")
    fig.savefig(out, dpi=110)
    print("wrote", out)

    # half-power beamwidths from the cuts
    def hpbw(arr: np.ndarray) -> float:
        peak = np.nanmax(arr)
        idx = int(np.nanargmax(arr))
        left = idx
        while left > 0 and arr[left] > peak - 3:
            left -= 1
        right = idx
        while right < len(arr) - 1 and arr[right] > peak - 3:
            right += 1
        return float(theta[right] - theta[left])

    print(f"maxGain={ff['maxGainDbi']:.2f} dBi  maxDir={ff['maxDirectivityDbi']:.2f} dBi  "
          f"eff={ff['radiationEfficiency']:.3f}")
    print(f"HPBW E-plane={hpbw(e_plane):.1f} deg  H-plane={hpbw(h_plane):.1f} deg")


if __name__ == "__main__":
    path = sys.argv[1]
    if "s11" in path or "tune" in path:
        plot_s11(path)
    else:
        plot_pattern(path)
