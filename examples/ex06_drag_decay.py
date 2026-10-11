"""Example 6 - atmospheric drag and orbital lifetime of CubeSats.

Integrates two-body gravity + J2 + drag (piecewise-exponential atmosphere,
co-rotating with the Earth) until re-entry at 100 km, for a 3U CubeSat in
two attitudes. The ballistic coefficient Cd*A/m sets how fast the orbit
decays, which is why deorbit devices work by increasing area.
"""

import _common  # noqa: F401
from _common import img

import matplotlib.pyplot as plt
import numpy as np

from orbital_sim import AtmosphericDrag, J2Perturbation, OrbitalElements, coe_to_rv, propagate
from orbital_sim import plotting
from orbital_sim.constants import DAY

START_ALT = 300.0
CASES = [  # name, mass kg, area m^2
    ("3U CubeSat, high drag (0.034 m^2)", 4.0, 0.034),
    ("3U CubeSat, low drag (0.010 m^2)", 4.0, 0.010),
    ("6U CubeSat, 12 kg (0.022 m^2)", 12.0, 0.022),
]


def run(verbose=True):
    el = OrbitalElements.circular(START_ALT, i_deg=51.6)
    r0, v0 = coe_to_rv(el)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (name, m, area) in enumerate(CASES):
        drag = AtmosphericDrag(cd=2.2, area_m2=area, mass_kg=m)
        traj = propagate(r0, v0, 400 * DAY, dt=300.0, perturbations=[J2Perturbation(), drag],
                         rtol=1e-9, atol=1e-9)
        lifetime = traj.t[-1] / DAY
        reentered = traj.altitude[-1] <= 100.5
        rows.append({"case": name, "ballistic": drag.ballistic, "lifetime_days": lifetime,
                     "reentered": bool(reentered), "orbits": traj.t[-1] / el.period})
        ax.plot(traj.t / DAY, traj.altitude, color=plotting.COLORS[k], lw=0.8,
                label=f"{name}: {lifetime:.1f} d")
    ax.axhline(100, color="k", ls=":", lw=0.8)
    ax.set_xlabel("time [days]"); ax.set_ylabel("altitude [km]")
    ax.set_title(f"Orbital decay from {START_ALT:.0f} km (J2 + drag)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plotting.save(fig, img("06_drag_decay.png"))

    if verbose:
        print("=== Example 6: drag decay ===")
        for r in rows:
            status = "re-entered" if r["reentered"] else "still in orbit"
            print(f"{r['case']:<38} B={r['ballistic']:.4f} m^2/kg  lifetime {r['lifetime_days']:6.1f} days "
                  f"({r['orbits']:.0f} orbits, {status})")
    return {"rows": rows, "start_alt": START_ALT}


if __name__ == "__main__":
    run()
