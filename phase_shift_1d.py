#!/usr/bin/env python3
#
# Author: Raphael Lehner 2024
#
# One dimensional Lüscher formalism
#
import masterthesis as mt
import numpy as np

# lattice
a = 0.01
N = 50
L = a * N
print(f"N = {N}, L = {L}")

# mass
m = 0.5
print(f"m = {m}")

# potential
g = -10.0
print(f"V(x) = {g} \delta(x)")


def V(x, g=-10.0):
    if abs(x - 0.0 * a) < a / 2.0:
        return g / a
    return 0.0


lp_delta = mt.luescher.luescher_1d.lattice_potential(N, a, V)
k, delta = lp_delta.k_delta(m, order=4)

N = 4
print("\nPhase shift (even):")
# one dimensional scattering on delta potential: https://doi.org/10.1119/1.1371011
for ki, di in zip(k[1:2*N+1:2], delta[1:2*N+1:2]):
    print(
        f"k = {ki:.4f}: Lüscher = {di:.4f} vs. analytic = {np.arctan(-g * m / ki):.4f}"
    )
    
print("...\n\nPhase shift (odd):")
for ki, di in zip(k[2:2*N+2:2], delta[2:2*N+2:2]):
    print(f"k = {ki:.4f}: Lüscher = {di:.4f} vs. analytic = {0.0}")
print("...")
