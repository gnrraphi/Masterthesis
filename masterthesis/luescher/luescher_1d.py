#!/usr/bin/env python3
#
# Author: Raphael Lehner 2024
#
# One dimensional Lüscher formalism
#
import numpy as np
from scipy.optimize import fsolve


# free lattice
class lattice:
    def __init__(self, N, a=1.0):
        self.N = N
        self.a = a

        self.L = a * np.arange(N)
        self.K = 2.0 * np.pi / a / N * np.arange(N)

        self.E_0_dic = {}
        self.p_0_dic = {} # quasi-momentum
        self.k_0_dic = {} # E = k_0 ** 2 / 2 / m
        self.delta_0_dic = {}

    def fft(self, f):
        x = np.zeros(self.N, dtype=np.complex128)
        for i, ki in enumerate(self.K):
            for xj in self.L:
                x[i] += np.exp(-1.0j * xj * ki) * f(xj)
        return x * self.a

    def ifft(self, f):
        x = np.zeros(self.N, dtype=np.complex128)
        for i, xi in enumerate(self.K):
            for kj in self.L:
                x[i] += np.exp(1.0j * xi * kj) * f(kj)
        return x / self.N

    def H_0(self, m, order=2):
        assert order in [2, 4], f"Order {order} not implemented"
        mat = np.zeros([self.N, self.N])

        if order == 2:
            assert self.N >= 2
            x = -1.0 / (2.0 * m * self.a ** 2.0)
            for i in range(self.N):
                mat[i, i] -= 2.0 * x
            for i in range(self.N - 1):
                mat[i+1, i] += x
                mat[i, i+1] += x

            # periodic boundary conditions
            mat[0, -1] += x
            mat[-1, 0] += x

        if order == 4:
            assert self.N >= 3
            x = -1.0 / (24.0 * m * self.a ** 2.0)
            for i in range(self.N):
                mat[i, i] -= 30.0 * x
                mat[(i+1)%self.N, i] += 16.0 * x
                mat[i, (i+1)%self.N] += 16.0 * x
                mat[(i+2)%self.N, i] -= x
                mat[i, (i+2)%self.N] -= x
        return mat

    def _E_0(self, m, order=2):
        if (m, order) not in self.E_0_dic:
            self.E_0_dic[(m, order)] = np.linalg.eigvalsh(self.H_0(m, order))

    def E_0(self, m, order=2):
        self._E_0(m, order)
        return self.E_0_dic[(m, order)]

    def _p_0(self, m, order=2):
        # for order 4 no exact solution - maybe implement approx
        assert order in [2], f"Order {order} not implemented"

        if (m, order) not in self.p_0_dic:
            if (m, order) not in self.E_0_dic:
                self._E_0(m, order)

            if order == 2:
                self.p_0_dic[(m, order)] = np.arccos(
                # abs to avoid negative arguments close to zero
                    np.sqrt(
                        1.0 - m * self.a ** 2.0 / 2.0
                        * abs(self.E_0_dic[(m, order)])
                    )
                ) * 2.0 / self.a

    def p_0(self, m, order=2):
        self._p_0(m, order)
        return self.p_0_dic[(m, order)]

    def delta(self, m, E):
        L = self.a * self.N

        def cot(x):
            return 1.0 / np.tan(x)

        def inv_cot(x):
            return np.arctan(1.0 / x)

        def S(x):
            return -cot(np.pi * np.sqrt(x)) * np.pi / np.sqrt(x)

        d = inv_cot(
            S(2.0 * m * E * (L / 2.0 / np.pi) ** 2.0)
            / 2.0 / np.pi ** 2.0 * np.sqrt(2.0 * m * E) * L
        )
        return d
            
    def _k_delta_0(self, m, order=2):
        if (m, order) not in self.delta_0_dic:
            E = np.where(self.E_0(m, order)>=0.0, self.E_0(m, order), np.nan)
            k = np.sqrt(2.0 * m * E)
            self.k_0_dic[(m, order)] = k
            self.delta_0_dic[(m, order)] = np.array(
                [self.delta(m, ei) for ei in E]
            )

    def k_delta_0(self, m, order=2):
        self._k_delta_0(m, order)
        return self.k_0_dic[(m, order)], self.delta_0_dic[(m, order)]


# lattice with potential V
class lattice_potential(lattice):
    def __init__(self, N, a, V):
        super().__init__(N, a)
        self.V = V
        self.E_dic = {}
        self.p_dic = {}
        self.psi_dic = {}
        self.k_dic = {}
        self.delta_dic = {}
        self.sigma_dic = {}

    def H(self, m, order=2):
        return self.H_0(m, order) + np.diag([self.V(li) for li in self.L])

    def _E(self, m, order=2):
        if (m, order) not in self.E_dic:
            self.E_dic[(m, order)] = np.linalg.eigvalsh(self.H(m, order))

    def E(self, m, order=2):
        self._E(m, order)
        return self.E_dic[(m, order)]

    def _psi(self, m, order=2):
        if (m, order) not in self.psi_dic:
            x, y = np.linalg.eigh(self.H(m, order))
            self.E_dic[(m, order)] = x
            self.psi_dic[(m, order)] = y.T

    def _p(self, m, order=2):
        if (m, order) not in self.p_dic:
            self._E(m, order)
            V_mat = np.zeros([self.N, self.N], dtype=np.complex128)
            V_ifft = self.ifft(self.V)
            for i in range(self.N):
                for j in range(self.N):
                    V_mat[i, j] = V_ifft[(i-j)%self.N]
            def det(*x):
                y = np.zeros(self.N)
                z = np.linalg.det(V_mat - np.diag(self.E_dic[(m, order)]) + np.diag(x))
                y[0] = z.real
                y[1] = z.imag
                return y
            sol = fsolve(det, self.E_0(m, order))
            self.p_dic[(m, order)] = np.arccos(
                        np.sqrt(1.0 - m * self.a ** 2.0 / 2.0 * np.abs(sol))
                               ) * 2.0 / self.a

    def _k_delta(self, m, order=2):
        if (m, order) not in self.delta_dic:
            E = np.where(self.E(m, order)>=0.0, self.E(m, order), np.nan)
            k = np.sqrt(2.0 * m * E)
            self.k_dic[(m, order)] = k
            self.delta_dic[(m, order)] = np.array(
                [self.delta(m, ei) for ei in E]
            )

    # phase shift 
    def k_delta(self, m, order=2):
        self._k_delta(m, order)
        return self.k_dic[(m, order)], self.delta_dic[(m, order)]

    def _sigma(self, m, order):
        k, delta = self.k_delta(m, order)
        s = 2.0 * np.sin(delta) ** 2.0
        self.sigma_dic[(m, order)] = s
        
    # cross section
    def k_sigma(self, m, order=2):
        self._sigma(m, order)
        return self.k_dic[(m, order)], self.sigma_dic[(m, order)]
