#!/usr/bin/env python3
#
# Author: Raphael Lehner 2024
#
import gpt as g
from gpt.core.group import differentiable_functional

# linear sigma model

# S[phi] = -2 * kappa * sum_x,n,mu phi_n(x)^dag * phi_n(x+mu)
#          + \sum_x phi2(x) + lambda / 4! * sum_x (phi2(x) - 1)^2
#          - alpha * \sum_x phi_0(x)
# phi2(x) = sum_n |phi_n(x)|^2
# sum_x (phi2(x) - 1)^2 = sum_x |phi2(x)|^2 - 2 sum_x phi2(x) + vol
class linear_sigma_model(differentiable_functional):
    def __init__(self, kappa, l, alpha):
        self.kappa = kappa
        self.l = l
        self.alpha = alpha

        self.__name__ = f"linear_sigma_model({self.kappa},{self.l},{self.alpha})"

    def kappa_to_mass2(self, k, l, D):
        return (1.0 - 2.0 * l / 24.0) / k - 2.0 * D

    def kappa_to_lambda(self, k, l):
        return l / (2.0 * k) ** 2.0

    def kappa_to_alpha(self, k, a):
        return a / (2.0 * k) ** 0.5

    def __call__(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)

        act = -2.0 * self.kappa * g.inner_product(J, phi).real

        p2 = g.norm2(phi)
        act += p2

        if self.l != 0.0:
            phi2 = g.real(phi.grid)
            # TO DO: replace with g.adj(phi) * phi
            phi2 @= g.trace(phi * g.adj(phi))

            p4 = g.norm2(phi2)
            act += self.l / 24.0 * (p4 - 2.0 * p2 + phi.grid.fsites)

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vreal([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            act -= g.inner_product(J, phi).real

        return act

    @differentiable_functional.single_field_gradient
    def gradient(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)
            J += g.cshift(phi, mu, -1)

        frc = g.lattice(phi)
        frc @= -2.0 * self.kappa * J

        frc += 2.0 * phi

        if self.l != 0.0:
            phi2 = g.real(phi.grid)
            # TO DO: replace with g.adj(phi) * phi
            phi2 @= g.trace(phi * g.adj(phi))

            frc += self.l / 6.0 * phi2 * phi
            frc -= self.l / 6.0 * phi

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vcomplex([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            frc -= J

        return frc


# non-linear sigma model

# S[phi] = -2.0 * kappa * sum_x,n,mu phi_n(x)^dag * phi_n(x+mu)
#          - alpha * \sum_x phi_0(x)
# with sum_n |phi_n(x)|^2 = 1
class non_linear_sigma_model(differentiable_functional):
    def __init__(self, kappa, alpha):
        self.kappa = kappa
        self.alpha = alpha

        self.__name__ = f"non_linear_sigma_model({self.kappa},{self.alpha})"

    def __call__(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)

        act = -2.0 * self.kappa * g.inner_product(J, phi).real

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vreal([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            act -= g.inner_product(J, phi).real

        return act

    @differentiable_functional.single_field_gradient
    def gradient(self, phi):
        J = g.lattice(phi)
        J[:] = 0.0
        for mu in range(phi.grid.nd):
            J += g.cshift(phi, mu, +1)
            J += g.cshift(phi, mu, -1)

        frc = g.lattice(phi)
        frc @= -2.0 * self.kappa * J

        if self.alpha != 0.0:
            N = J.otype.shape[0]
            J_comp = g.vcomplex([self.alpha] + [0] * (N - 1), N)
            J[:] = J_comp
            frc -= J

        frc -= g.trace(frc * g.adj(phi)) * phi

        return frc

    # https://arxiv.org/abs/1102.1852
    def constrained_leap_frog(self, eps, phi, mom_phi):
        # TO DO: replace with g.adj(v1) * v2
        def dot(v1, v2):
            return g.trace(v2 * g.adj(v1))

        n = g.real(phi.grid)
        n @= g.component.sqrt(g.component.real(dot(mom_phi, mom_phi)))

        # phi'      =  cos(alpha) phi + (1/|mom_phi|) sin(alpha) mom_phi
        # mom_phi'  = -|mom_phi| sin(alpha) phi + cos(alpha) mom_phi
        # alpha = eps |mom_phi|
        _phi = g.lattice(phi)
        _phi @= phi

        cos = g.real(phi.grid)
        cos @= g.component.cos(eps * n)

        sin = g.real(phi.grid)
        sin @= g.component.sin(eps * n)

        phi @= cos * _phi + g(g.component.inv(n) * sin) * mom_phi
        mom_phi @= -g(n * sin) * _phi + cos * mom_phi
        del _phi, cos, sin, n

    # https://arxiv.org/abs/1102.1852
    def draw(self, field, rng, constraint=None):
        if constraint is None:
            phi = field
            rng.element(phi)
            n = g.component.real(g.trace(phi * g.adj(phi)))
            phi @= phi * g.component.inv(g.component.sqrt(n))
        else:
            mom_phi = field
            phi = constraint
            rng.normal_element(mom_phi)
            mom_phi @= mom_phi - g(phi * g.adj(phi)) * mom_phi
