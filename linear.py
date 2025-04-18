"""
Linear Convection Code adapted from Glatzmaier 2014

Usage:
    linear.py [options]

Options:
 --Ra=<Ra>      # Rayleigh number [default: 600]
 --Pr=<Pr>      # Prandtl number [default: 1]
 --Nz=<Nz>      # z-Resolution [default: 128]
 --Nn=<Nn>      # Fourier modes [default: 10]
 --a=<a>        # Aspect Ratio [default: 1]
 --stop=<stop>  # Sim-time to stop at [default: 0.5]
 --crate=<crate>    # Check rate [default: 100]
 -o <OUT>           # Output directory [default: ./DATA/]
"""

import numpy as np
from docopt import docopt
from tridiagonal import tridiagonal
from pathlib import Path

args = docopt(__doc__)
print(args)
Nz = int(args["--Nz"])
Nn = int(args["--Nn"])
outdir = Path(args["-o"])
outdir.mkdir(exist_ok=True)
# Arrays to hold tridiagonal matrix information
sub = np.zeros(Nz)
dia = np.zeros(Nz)
sup = np.zeros(Nz)
wk1 = np.zeros(Nz)
wk2 = np.zeros(Nz)

# Streamfunction, vorticity and temp fields
φ = np.zeros((Nz, Nn))
ω = np.zeros((Nz, Nn))
T = np.zeros((Nz, Nn))

# Arrays to store time derivatives of ω and T
dωdt = np.zeros((Nz, Nn, 2))
dTdt = np.zeros((Nz, Nn, 2))

# Spacing between z-levels
dz = 1.0 / (Nz - 1)
# constant to hold 1/dz^2
oodz2 = 1 / (dz * dz)
# constant to hold pi/a
c = np.pi / float(args["--a"])


# Populate sub and sup lines of matrix:
sub += -oodz2
sub[Nz - 1] = 0

sup += -oodz2
sup[0] = 0

Ra = float(args["--Ra"])
Pr = float(args["--Pr"])
t = 0
dt = 0.9 * (dz * dz) / 4
if Pr > 1:
    dt /= Pr

# ? Temp initial condition: T = sin(pi*z)
z = np.linspace(0, 1, Nz)
T[:, 0] = 1 - z
for n in range(1, Nn):
    T[:, n] = np.sin(np.pi * z)

check_rate = int(args["--crate"])
mid_depth_idx = Nz // 2

stop_time = float(args["--stop"])
i = 0
# ! Perform Sim Loop!
while t <= stop_time:
    #! Check if amplitudes are growing or decaying
    if i % check_rate == 0:
        print(f"Loop {i:>6}, t={dt*i:.1e}")
        print(f" n | Temp Amp | Vort Amp | Strf Amp | Temp log | Vort Log | Strf Log |")
        current_temp_amp = np.log(np.abs(T[mid_depth_idx, :]))
        current_ω_amp = np.log(np.abs(ω[mid_depth_idx, :]))
        current_φ_amp = np.log(np.abs(φ[mid_depth_idx, :]))
        if i == 0:
            last_temp_amp = current_temp_amp
            last_ω_amp = current_ω_amp
            last_φ_amp = current_φ_amp
            for n in range(1, Nn):
                print(
                    f"{n:<3}|{T[mid_depth_idx, n]:<10.2e}|{ω[mid_depth_idx, n]:<10.2e}|{φ[mid_depth_idx, n]:<10.2e}|"
                )
            out_temp = T
            out_φ = φ
            out_ω = ω
            out_times = [dt * i]
        else:
            temp_check = current_temp_amp - last_temp_amp
            ω_check = current_ω_amp - last_ω_amp
            φ_check = current_φ_amp - last_φ_amp
            for n in range(0, Nn):
                print(
                    f"{n:<3}|{T[mid_depth_idx, n]:<10.2e}|{ω[mid_depth_idx, n]:<10.2e}|{φ[mid_depth_idx, n]:<10.2e}|{temp_check[n]:<10.2e}|{ω_check[n]:<10.2e}|{φ_check[n]:<10.2e}|"
                )
            last_temp_amp = current_temp_amp
            last_φ_amp = current_φ_amp
            last_ω_amp = current_ω_amp

            out_temp = np.dstack((out_temp, T))
            out_φ = np.dstack((out_φ, φ))
            out_ω = np.dstack((out_ω, ω))

            out_times.append(dt * i)

    # ! Compute Time Derivatives using vertical finite difference method
    for k in range(1, Nz - 1):
        for n in range(0, Nn):
            # dT/dt = (n pi / a)*φ_n + (d^2(T_n)/dz^2 - (n pi/a)^2 T_n)
            dTdt[k, n, 1] = (n * c) * φ[k, n] + (
                (oodz2 * (T[k + 1, n] - 2 * T[k, n] + T[k - 1, n]))
                - ((n * c) ** 2 * T[k, n])
            )
            # dω/dt = Ra/Pr * (n*pi/a) * T_n + (d^2(ω_n)/dz^2 - (n pi/a)^2 ω_n)
            dωdt[k, n, 1] = (Ra / Pr) * (n * c) * T[k, n] + (
                (oodz2 * (ω[k + 1, n] - 2 * ω[k, n] + ω[k - 1, n]))
                - ((n * c) ** 2 * ω[k, n])
            )

    #! Update T, ω to new values via Adams-Bashforth timestepping
    for k in range(0, Nz):
        for n in range(0, Nn):
            T[k, n] += (dt / 2) * (3 * (dTdt[k, n, 1]) - (dTdt[k, n, 0]))
            ω[k, n] += (dt / 2) * (3 * (dωdt[k, n, 1]) - (dωdt[k, n, 0]))

    #! Update psi using tridiagonal solver
    for n in range(0, Nn):
        for k in range(0, Nz - 1):
            if k == 0:
                dia[k] = 1
            else:
                dia[k] = (n * c) ** 2 + 2 * oodz2
        dia[-1] = 1
        φ[:, n] = tridiagonal(sub, dia, sup, ω[:, n])

    #! Update previous dTdt and dωdt to new values
    for n in range(0, Nn):
        for k in range(0, Nz):
            dTdt[k, n, 0] = dTdt[k, n, 1]
            dωdt[k, n, 0] = dωdt[k, n, 1]

    #! Update t and i
    i += 1
    t += dt


# ! Final Output
print(f"Loop {i:0>4}, t={dt*i:.1e}")
print(f" n | Temp Amp | Vort Amp | Strf Amp | Temp log | Vort Log | Strf Log |")
current_temp_amp = np.log(np.abs(T[mid_depth_idx, :]))
current_ω_amp = np.log(np.abs(ω[mid_depth_idx, :]))
current_φ_amp = np.log(np.abs(φ[mid_depth_idx, :]))
temp_check = current_temp_amp - last_temp_amp
ω_check = current_ω_amp - last_ω_amp
φ_check = current_φ_amp - last_φ_amp
for n in range(0, Nn):
    print(
        f"{n:<3}|{T[mid_depth_idx, n]:<10.2e}|{ω[mid_depth_idx, n]:<10.2e}|{φ[mid_depth_idx, n]:<10.2e}|{temp_check[n]:<10.2e}|{ω_check[n]:<10.2e}|{φ_check[n]:<10.2e}|"
    )
last_temp_amp = current_temp_amp
last_φ_amp = current_φ_amp
last_ω_amp = current_ω_amp

np.savez(
    outdir.joinpath(f"{Ra:3>0.0f}_Tamps"),
    times=out_times,
    T_amps=out_temp,
    φ_amps=out_φ,
    ω_amps=out_ω,
)
