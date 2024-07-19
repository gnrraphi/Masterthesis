#!/usr/bin/env python3
#
# Author: Raphael Lehner 2024
#
# Pion mass from non-linear sigma model
#
import masterthesis as mt
import gpt as g
import numpy as np
import gvar as gv

# lattice
g.default.set_verbose("step_size", False)
L, T = 8, 16
grid = g.grid([L, L, L, T], g.double)
g.message(f"Lattice = {grid.fdimensions}")
rng = g.random("hmc-nlsm")

# action
kappa = 0.315
alpha = 0.010
g.message("Actions:")
a0 = g.qcd.scalar.action.mass_term()
a1 = mt.sigma_model.non_linear_sigma_model(kappa, alpha)
g.message(f" - {a0.__name__}")
g.message(f" - {a1.__name__}")

# field
N = 4
phi = g.vreal(grid, N)
a1.draw(phi, rng)

# conjugate momenta
mom = g.group.cartesian(phi)

# molecular dynamics
sympl = g.algorithms.integrator.symplectic

ip = sympl.update_p(mom, lambda: a1.gradient(phi, phi))


class constrained_iq(sympl.symplectic_base):
    def __init__(self, phi, mom_phi):
        def inner(eps):
            a1.constrained_leap_frog(eps, phi, mom_phi)

        super().__init__(1, [], [inner], None, "constrained iq")


iq = constrained_iq(phi, mom)

# integrator
md_step = 8
mdint = sympl.leap_frog(md_step, ip, iq)
g.message(f"Integration scheme:\n{mdint}")

# metropolis
metro = g.algorithms.markov.metropolis(rng)

# MD units
tau = 0.5
g.message(f"tau = {tau} MD units")


def hamiltonian():
    return a0(mom) + a1(phi)


def hmc(tau):
    a1.draw(mom, rng, phi)

    accrej = metro(phi)
    h0 = hamiltonian()
    mdint(tau)
    h1 = hamiltonian()
    return [accrej(h1, h0), h1 - h0]


# thermalization
for i in range(1, 6):
    h = []
    for j in range(50):
        h += [hmc(tau)]

    h = np.array(h)
    g.message(f"{i*20} % of thermalization completed")
    g.message(
        f"Action = {a1(phi)}, Acceptance = {np.mean(h[:,0]):.2f}, |dH| = {np.mean(np.abs(h[:,1])):.4e}")


def measure(phi):
    vol = L ** 3
    phit = g.slice(phi / vol, 3)
    return np.array([
        np.mean([
            phit[(t+s)%T][a] * phit[s][a].conjugate()
            for a in range(1, 4) for s in range(T)]
        ).real for t in range(T)
    ])


c2_pi = []

# production
for i in range(1, 11):
    h = []
    for j in range(50):

        # sub-steps
        for k in range(2):
            h += [hmc(tau)]

        c2_pi += [measure(phi)]

    h = np.array(h)
    g.message(f"{i*10} % of production completed")
    g.message(
        f"Action = {a1(phi)}, Acceptance = {np.mean(h[:,0]):.2f}, |dH| = {np.mean(np.abs(h[:,1])):.4e}"
    )

# pion mass fit
c2_pi_avg = gv.dataset.avg_data(c2_pi)
m_eff = np.mean(gv.log(c2_pi_avg / np.roll(c2_pi_avg, -1))[:3])
g.message(f"m_pi = {m_eff}")