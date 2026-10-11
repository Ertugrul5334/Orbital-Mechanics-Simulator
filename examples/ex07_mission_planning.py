"""Example 7 - multi-orbit mission planning: getting to GEO from Cape Canaveral.

A 2,000 kg spacecraft (Isp 320 s) starts in a 300 km, 28.5 deg parking orbit
and must reach an equatorial geostationary orbit. Three strategies are flown
through the full numerical propagator (with J2), and the delta-v and
propellant budgets are compared:

  A. remove all 28.5 deg of inclination in LEO, then a coplanar Hohmann
  B. Hohmann with the whole plane change combined into the apogee burn
  C. optimal split: a small plane change at perigee, the rest at apogee
"""

import _common  # noqa: F401
from _common import img

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from orbital_sim import (CoastToNode, Coast, Hohmann, J2Perturbation, Mission, OrbitalElements,
                         PlaneChange, R_EARTH, combined_plane_change_dv)
from orbital_sim import plotting
from orbital_sim.maneuvers import circular_speed, vis_viva

R_LEO = R_EARTH + 300.0
R_GEO = 42164.0
I0 = 28.5


def analytic_split_cost(di1_deg):
    """Total dv [km/s] when di1 deg is removed at perigee and the rest at apogee."""
    a_t = 0.5 * (R_LEO + R_GEO)
    v1, vp = circular_speed(R_LEO), vis_viva(R_LEO, a_t)
    va, v2 = vis_viva(R_GEO, a_t), circular_speed(R_GEO)
    return (combined_plane_change_dv(v1, vp, math.radians(di1_deg))
            + combined_plane_change_dv(va, v2, math.radians(I0 - di1_deg)))


def new_mission():
    m = Mission(OrbitalElements.circular(300.0, i_deg=I0, raan_deg=0.0, u_deg=45.0),
                mass_kg=2000.0, isp_s=320.0, perturbations=[J2Perturbation()], output_dt=60.0)
    return m.add(Coast(2 * 5430.0))       # two parking-orbit revolutions for checkout


def run(verbose=True):
    best = minimize_scalar(analytic_split_cost, bounds=(0, I0), method="bounded")
    di1 = float(best.x)

    # J2 drifts the node/apogee a little during the 5 h transfer, so B and C
    # finish with a small residual inclination that a trim burn removes.
    trim = PlaneChange(0.0, name="Inclination trim at GEO node")
    strategies = {
        "A: plane change in LEO, then Hohmann":
            new_mission().add(PlaneChange(0.0)).add(Hohmann(R_GEO)),
        "B: plane change combined with apogee burn":
            new_mission().add(CoastToNode()).add(Hohmann(R_GEO, final_inclination_deg=0.0)).add(trim),
        f"C: optimal split ({di1:.2f} deg at perigee)":
            new_mission().add(CoastToNode()).add(Hohmann(R_GEO, final_inclination_deg=0.0,
                                                         transfer_inclination_deg=I0 - di1)).add(trim),
    }
    results = {}
    for name, mission in strategies.items():
        mission.add(Coast(86400.0))          # one day on station to confirm the final orbit
        results[name] = mission.run()

    # Plot strategy C in 3D and the split trade-off curve.
    resC = results[list(results)[2]]
    labels = ["parking orbit", "coast to node", "GTO (transfer)", "GEO: coast to node", "GEO (1 day, J2)"]
    ax = plotting.plot_orbits_3d(resC.segments, labels[:len(resC.segments)],
                                 burns=[resC.trajectory.r[np.searchsorted(resC.trajectory.t, b.time)]
                                        for b in resC.burns],
                                 title="Mission C: LEO 28.5 deg -> GEO")
    ax.view_init(elev=25, azim=-50)
    plotting.save(ax, img("07_mission_3d.png"))

    split = np.linspace(0, I0, 200)
    fig, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(split, [analytic_split_cost(s) * 1000 for s in split], color=plotting.COLORS[0])
    ax2.axvline(di1, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("inclination removed at perigee [deg]"); ax2.set_ylabel("total dv [m/s]")
    ax2.set_title(f"Optimal plane-change split: {di1:.2f} deg at perigee"); ax2.grid(alpha=0.3)
    plotting.save(fig, img("07_plane_change_split.png"))

    names = list(results)
    fig, ax3 = plt.subplots(figsize=(8, 4.2))
    dvs = [results[n].total_dv * 1000 for n in names]
    ax3.barh([n.split(":")[0] for n in names], dvs, color=plotting.COLORS[:3])
    for k, (n, dv) in enumerate(zip(names, dvs)):
        ax3.text(dv + 50, k, f"{dv:,.0f} m/s, {results[n].total_propellant:,.0f} kg propellant", va="center")
    ax3.set_xlim(0, max(dvs) * 1.55); ax3.set_xlabel("total dv [m/s]")
    ax3.set_title("LEO (28.5 deg) to GEO: delta-v budget by strategy"); ax3.invert_yaxis()
    plotting.save(fig, img("07_strategy_dv.png"))

    if verbose:
        print("=== Example 7: mission planning, LEO 28.5 deg -> GEO ===")
        for name, res in results.items():
            print(f"\n--- {name}")
            print(res.summary())
        print(f"\nAnalytic optimum split: {di1:.3f} deg at perigee, total {best.fun * 1000:.1f} m/s")
    return {"results": results, "optimal_split_deg": di1, "optimal_dv": best.fun}


if __name__ == "__main__":
    run()
