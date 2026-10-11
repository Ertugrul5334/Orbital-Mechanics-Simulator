"""Example 5 - J2 (Earth oblateness) perturbation and sun-synchronous orbits.

Integrates the full J2 acceleration for 10 days and compares the simulated
drift of RAAN and argument of perigee with first-order secular theory:

    RAAN_dot = -1.5 n J2 (R/p)^2 cos i

A 700 km sun-synchronous orbit (i ~ 98.19 deg) should precess by
+0.9856 deg/day, following the Sun. The ISS orbit regresses about -5 deg/day.
"""

import _common  # noqa: F401
from _common import img

import math

import matplotlib.pyplot as plt
import numpy as np

from orbital_sim import (J2Perturbation, OrbitalElements, R_EARTH, coe_to_rv, j2_secular_rates,
                         propagate, sun_synchronous_inclination)
from orbital_sim import plotting
from orbital_sim.constants import DAY, SUN_SYNC_RAAN_RATE


def drift(el, days):
    r, v = coe_to_rv(el)
    traj = propagate(r, v, days * DAY, dt=600.0, perturbations=[J2Perturbation()], rtol=1e-10, atol=1e-10)
    hist = traj.element_history()
    t_days = hist["t"] / DAY
    raan_rate = np.polyfit(t_days, hist["raan"], 1)[0]
    argp = None
    if el.e > 1e-3:
        argp = np.polyfit(t_days, hist["argp"], 1)[0]
    return traj, hist, t_days, raan_rate, argp


def run(verbose=True):
    days = 10
    a_sso = R_EARTH + 700
    i_sso = sun_synchronous_inclination(a_sso)
    cases = {
        "SSO 700 km": OrbitalElements(a_sso, 0.0, i_sso, 0.0, 0.0, 0.0),
        "ISS 420 km, 51.64 deg": OrbitalElements.circular(420, i_deg=51.64),
        "Molniya-type, 63.4 deg": OrbitalElements.from_degrees(26600, 0.74, 63.4, 0, 270, 0),
        "Eccentric LEO, 30 deg": OrbitalElements.from_degrees(R_EARTH + 1500, 0.1, 30.0, 0, 0, 0),
    }
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (name, el) in enumerate(cases.items()):
        traj, hist, t_days, raan_sim, argp_sim = drift(el, days)
        raan_th, argp_th = (math.degrees(x) * DAY for x in j2_secular_rates(el.a, el.e, el.i))
        rows.append({"case": name, "i_deg": math.degrees(el.i), "raan_sim": raan_sim, "raan_theory": raan_th,
                     "raan_err_pct": 100 * abs(raan_sim - raan_th) / abs(raan_th),
                     "argp_sim": argp_sim, "argp_theory": argp_th})
        ax.plot(t_days, hist["raan"] - hist["raan"][0], color=plotting.COLORS[k], label=f"{name} (sim)")
        ax.plot(t_days, raan_th * t_days, ls="--", color=plotting.COLORS[k], lw=0.9)
    ax.set_xlabel("time [days]"); ax.set_ylabel("change in RAAN [deg]")
    ax.set_title("Nodal precession from J2: simulation (solid) vs theory (dashed)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plotting.save(fig, img("05_j2_raan_drift.png"))

    # SSO inclination vs altitude.
    alts = np.linspace(200, 1500, 100)
    incl = [math.degrees(sun_synchronous_inclination(R_EARTH + h)) for h in alts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alts, incl, color=plotting.COLORS[0])
    ax.set_xlabel("altitude [km]"); ax.set_ylabel("sun-synchronous inclination [deg]")
    ax.set_title("Inclination needed for 0.9856 deg/day nodal precession"); ax.grid(alpha=0.3)
    plotting.save(fig, img("05_sso_inclination.png"))

    out = {"rows": rows, "sso_incl_deg": math.degrees(i_sso),
           "sso_required_deg_day": math.degrees(SUN_SYNC_RAAN_RATE) * DAY, "days": days}
    if verbose:
        print("=== Example 5: J2 perturbation ===")
        print(f"SSO at 700 km needs i = {out['sso_incl_deg']:.3f} deg for "
              f"{out['sso_required_deg_day']:.4f} deg/day")
        print(f"{'case':<26}{'i deg':>8}{'RAAN sim':>11}{'theory':>9}{'err %':>8}{'argp sim':>10}{'theory':>9}  [deg/day]")
        for r in rows:
            ap = f"{r['argp_sim']:>10.4f}{r['argp_theory']:>9.4f}" if r["argp_sim"] is not None else f"{'-':>10}{'-':>9}"
            print(f"{r['case']:<26}{r['i_deg']:>8.3f}{r['raan_sim']:>11.4f}{r['raan_theory']:>9.4f}"
                  f"{r['raan_err_pct']:>8.2f}{ap}")
    return out


if __name__ == "__main__":
    run()
