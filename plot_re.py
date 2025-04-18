"""
Code to plot the Reynolds number against time. If Reynolds number
increases the system is growing and supercritical. If Reynolds number
decreases the system is decaying and subcritical.

Usage:
    plot_re.py <direc> <imname>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from docopt import docopt
from pathlib import Path
import re

args = docopt(__doc__)

direc = Path(args["<direc>"])
runfiles = sorted(
    list(direc.glob("*.npz")), key=lambda f: int(re.sub(r"\D", "", f.name))
)
Re_dict = {}
for runfile in runfiles:
    Ra = int(re.sub(r"\D", "", runfile.name))
    print(f"Ra = {Ra}")
    data = np.load(runfile)
    times = data["times"]
    T_amps = data["T_amps"]
    φ_amps = data["φ_amps"]

    T_amps.shape
    Nmodes = T_amps.shape[1]
    Nz = T_amps.shape[0]
    T = T_amps[:, :, 0]
    Nx = 64
    a = 5
    x = np.linspace(0, a, Nx)
    z = np.linspace(0, 1, Nz)
    c = np.pi / a
    T0x = np.zeros(Nmodes)
    Tz = np.zeros(x.shape)
    T_field = np.zeros((Nz, Nx))
    T_aves = []
    θ_aves = []
    ptimes = []
    Res = []
    for tidx, tval in enumerate(times):
        for zidx, zval in enumerate(z):
            for xidx, xval in enumerate(x):
                for n in range(Nmodes):
                    T0x[n] = φ_amps[zidx, n, tidx] * np.sin(n * c * xval)
                Tz[xidx] = np.sum(T0x)
            T_field[zidx, :] = Tz
        T_ave = np.trapezoid(np.trapezoid(T_field, x=z, axis=0), x=x, axis=0)
        T_aves.append(T_ave)
        ptimes.append(tval)
        φ_field = T_field
        u, w = np.gradient(φ_field, z, x)

        u_mag = np.sqrt(u * u + w * w)
        Re = (1 / Nx) * np.trapezoid(np.trapezoid(u_mag, x=z, axis=0), x=x, axis=0)
        Res.append(Re)
    Re_dict[Ra] = Res

fig, ax = plt.subplots(1, 1)
for key in Re_dict.keys():
    ax.plot(ptimes, 5 * np.array(Re_dict[key]), label=f"Ra={key}")
ax.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")
ax.axhline(2.47, c="grey", ls="--", alpha=0.2)
ax.set_yscale("log")
ax.set_xlabel(r"Time, $\tau_\nu$")
ax.set_xlim(0)
ax.set_ylabel("Re")

plt.savefig(direc.joinpath(args["<imname>"] + ".pdf"), bbox_inches="tight")
