#!/usr/bin/env python3
#
# Author: Raphael Lehner 2024
#
# MCMC methods
#
import numpy as np
import scipy.integrate as integrate


# Metropolis-Hastings algorithm for \int_{-xlim}^{xlim}\mathrm{d}x\, f(x)p(x)
def mh_integrate(f, p, eps, N, xlim, verbose=False):
    x = 0.0
    ftot = []
    reject = 0
    accept = 0
    n = 0
    for i in range(N):
        x_new = np.random.normal(x, eps)
        if np.random.uniform(0.0, 1.0) < p(x_new) / p(x) and not (
            x_new < -xlim or x_new > xlim
        ):
            accept += 1
            x = x_new
        else:
            reject += 1
        ftot += [f(x)]
    if verbose:
        print("    Acceptance rate", accept / (accept + reject))
    return np.array(ftot)


# sample size
N = 1000000

# step size
eps = 1.2

# integral
xlim = 50.0

p0 = lambda x: 1.0
f0 = lambda x: x**2 * np.exp(-(x**2) / 2.0)
Z0 = integrate.quad(lambda x: p0(x), -xlim, xlim)[0]

p1 = lambda x: np.exp(-(x**2) / 2.0)
f1 = lambda x: x**2
Z1 = integrate.quad(lambda x: p1(x), -xlim, xlim)[0]

result = integrate.quad(lambda x: f0(x), -xlim, xlim)[0]
print(f"Metropolis-Hastings algorithm for \int_-{xlim}^{xlim} dx x^2 e^(-x^2/2)")
print(f"N = {N}")
print(f"Exact result: {result:.4f}")

print(f"eps = {eps}, p(x) \propto 1:")
farr0 = mh_integrate(f0, p0, eps, N, xlim, True) * Z0
print(f"    {np.mean(farr0):.4f} +- {np.std(farr0) / N**0.5:.4f}")

print(f"eps = {eps}, p(x) \propto e^(-x^2/2):")
farr1 = mh_integrate(f1, p1, eps, N, xlim, True) * Z1
print(f"    {np.mean(farr1):.4f} +- {np.std(farr1) / N**0.5:.4f}")


def V(q):
    return 1.0 / 2.0 * q**2


def dV(q):
    return q


def integrator_leapfrog(p, q, dt):
    p_half = p - dV(q) * dt / 2.0
    q_one = q + p_half * dt
    p_one = p_half - dV(q_one) * dt / 2.0
    return p_one, q_one


# HMC algorithm for \int_{-\infty}^{\infty}\mathrm{d}x\, f(x) e^{-x^2/2}
def hmc_integrate(f, eps, trajectory_length, trajectories, integrator, verbose=False):
    q = 0.0
    ftot = []
    dE2tot = 0.0
    reject = 0
    accept = 0
    for i in range(trajectories):
        p = np.random.normal()
        E0 = p**2.0 / 2.0 + V(q)
        q0 = q
        for j in range(trajectory_length):
            p, q = integrator(p, q, eps)
        E1 = p**2.0 / 2.0 + V(q)
        if np.exp(E0 - E1) < np.random.uniform(0.0, 1.0):
            reject += 1
            q = q0
        else:
            accept += 1
            dE2tot += (E1 - E0) ** 2.0
        ftot += [f(q)]
    if verbose:
        print(f"    Energy changed by {(dE2tot / trajectories) ** 0.5:.4f}")
        print(f"    Acceptance rate {accept / (accept + reject)}")
    return np.array(ftot)


# sample size
N = 1000000

# integral
xlim = np.infty

p = lambda x: np.exp(-(x**2) / 2.0)
f = lambda x: x**2
Z = integrate.quad(lambda x: p(x), -xlim, xlim)[0]

result = integrate.quad(lambda x: f0(x), -xlim, xlim)[0]
print(f"\nHMC for \int_-{xlim}^{xlim} dx x^2 e^(-x^2/2)")
print(f"N = {N}")
print(f"Exact result: {result:.4f}")

dt = 1.83
md = 3
print(f"dt = {dt}, MD steps = {md}:")
hmcarr = hmc_integrate(f, dt, md, N, integrator_leapfrog, True) * Z
print(f"    {np.mean(hmcarr):.4f} +- {np.std(hmcarr) / N**0.5:.4f}")

print(f"Metropolis for comparison:")
print(f"eps = {eps}, p(x) \propto e^(-x^2/2):")
farr2 = mh_integrate(f, p, 1.2, N, xlim, True) * Z
print(f"    {np.mean(farr2):.4f} +- {np.std(farr2) / N**0.5:.4f}")
