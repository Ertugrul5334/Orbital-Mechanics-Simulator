"""Example 1 - two-body propagation, element conversions, ground tracks, integrator accuracy.

Propagates the ISS orbit for one day, converts every state back to the six
classical elements, plots the 3D orbit and ground tracks, and measures the
error of the hand-written RK4 integrator against Kepler's analytic solution.
"""

import _common  # noqa: F401  (path + backend setup)
from _common import img

import math

import matplotlib.pyplot as plt
import numpy as np

from orbital_sim import OrbitalElements, coe_to_rv, kepler_propagate, propagate, rv_to_coe
from orbital_sim import plotting
from orbital_sim.groundtrack import ground_track_shift_deg


def run(verbose=True):
    iss = OrbitalElements.from_degrees(6378.137 + 420, 0.0005, 51.64, 30.0, 90.0, 0.0)
    r0, v0 = coe_to_rv(iss)
    duration = 86400.0
    traj = propagate(r0, v0, duration, dt=20.0, label="ISS-like orbit")

    # Round-trip all 6 elements at every sample: r,v -> COE -> r,v.
    worst_rt = 0.0
    for k in range(0, len(traj.t), 10):
        el = rv_to_coe(traj.r[k], traj.v[k])
        r_back, v_back = coe_to_rv(el)
        worst_rt = max(worst_rt, np.linalg.norm(r_back - traj.r[k]))

    # Accuracy against the analytic Kepler solution.
    r_exact = coe_to_rv(kepler_propagate(iss, traj.t[-1]))[0]
    err_dop = np.linalg.norm(traj.r[-1] - r_exact)
    energy = traj.energy()
    energy_drift = abs(energy[-1] - energy[0]) / abs(energy[0])

    # RK4 convergence study: position error after 1 orbit vs step size.
    steps = [120, 60, 30, 15, 7.5]
    rk4_err = []
    for h in steps:
        tr = propagate(r0, v0, iss.period, dt=h, method="rk4")
        rk4_err.append(np.linalg.norm(tr.r[-1] - coe_to_rv(kepler_propagate(iss, tr.t[-1]))[0]))
    order = np.polyfit(np.log(steps), np.log(rk4_err), 1)[0]

    # Figures
    plotting.save(plotting.plot_orbits_3d([traj], title="ISS-like orbit, 24 h (ECI)"), img("01_orbit_3d.png"))
    one_rev = propagate(r0, v0, 3 * iss.period, dt=20.0)
    plotting.save(plotting.plot_ground_track(one_rev, title="Ground track, 3 orbits (ISS-like)"),
                  img("01_ground_track.png"))
    plotting.save(plotting.plot_ground_track_3d(one_rev), img("01_ground_track_3d.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(steps, np.array(rk4_err) * 1000, "o-", color=plotting.COLORS[0], label="RK4 (this code)")
    ax.loglog(steps, rk4_err[0] * 1000 * (np.array(steps) / steps[0]) ** 4, "--", color="gray", label="slope 4")
    ax.axhline(err_dop / duration * iss.period * 1000, color=plotting.COLORS[1], ls=":", label="DOP853 (per orbit)")
    ax.set_xlabel("step size [s]"); ax.set_ylabel("position error after 1 orbit [m]")
    ax.set_title(f"RK4 convergence (fitted order {order:.2f})")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    plotting.save(fig, img("01_rk4_convergence.png"))

    out = {
        "duration_s": duration,
        "samples": len(traj.t),
        "period_min": iss.period / 60,
        "orbits": duration / iss.period,
        "roundtrip_err_m": worst_rt * 1000,
        "dop853_err_m": err_dop * 1000,
        "energy_drift_rel": energy_drift,
        "rk4_err_m": dict(zip(steps, np.array(rk4_err) * 1000)),
        "rk4_order": order,
        "gt_shift_deg": ground_track_shift_deg(iss.period),
        "initial_elements": str(iss),
        "final_elements": str(rv_to_coe(*traj.final_state)),
    }
    if verbose:
        print("=== Example 1: two-body propagation ===")
        print(f"Propagated {duration:.0f} s ({out['orbits']:.2f} orbits, {out['samples']} samples)")
        print(f"Initial: {out['initial_elements']}")
        print(f"Final:   {out['final_elements']}")
        print(f"COE round-trip error (worst):   {out['roundtrip_err_m']:.2e} m")
        print(f"DOP853 error vs Kepler (24 h):   {out['dop853_err_m']:.3f} m")
        print(f"Relative energy drift (24 h):    {out['energy_drift_rel']:.2e}")
        print(f"RK4 fitted convergence order:    {order:.2f}")
        for h, e in out["rk4_err_m"].items():
            print(f"   RK4 dt={h:>6} s -> {e:10.3f} m error after 1 orbit")
        print(f"Ground-track shift per orbit:    {out['gt_shift_deg']:.2f} deg west")
    return out


if __name__ == "__main__":
    run()
