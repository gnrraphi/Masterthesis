#!/usr/bin/env python3
#
# Author: Raphael Lehner 2024
#
import gpt as g
import numpy as np


class wolff_cluster:
    def __init__(self, rng, grid, kappa, alpha, verbose=True):

        self.rng = rng
        self.k = kappa
        self.a = alpha
        self.verbose = verbose

        coord = g.coordinates(grid)
        nd = grid.nd
        fdim = grid.fdimensions
        stride = [1]
        for dim in range(nd-1):
            stride += [stride[dim] * fdim[dim]]

        t0 = g.time()
        nn_idx = []
        for cx in coord:
            nn_ix = []
            for idx in range(nd):
                for sign in [-1, 1]:
                    cx_new = np.copy(cx)
                    cx_new[idx] = (cx_new[idx] + sign) % fdim[idx]
                    nn_ix += [np.sum([xi * si for xi, si in zip(cx_new, stride)])]
            nn_idx += [np.array(nn_ix)]
        self.coord = coord
        self.nn_idx = nn_idx
        t1 = g.time()

        if self.verbose:
            g.message(f"Wolff: Init nearest-neighbor in {t1-t0:.4f} s")

    def __call__(self, field, max_len=5000):
        t0 = g.time()
        N = field.otype.shape[0]
        r = g.tensor(
            np.array([[self.rng.normal() for i in range(N)]], dtype=np.complex128),
            field.otype
        )
        r /= np.sqrt((g.adj(r) * r).real)

        idx = self.rng.uniform_int(min=0, max=int(field.grid.fsites)-1)
        C = [idx]
        F = [idx]
        t1 = g.time()

        if self.verbose:
            g.message(f"Wolff: Init cluster in {t1-t0:.4f} s")

        t0 = g.time()
        while len(F) != 0:
            F_new = []
            for i in F:
                for j in self.nn_idx[i]:
                    if j in C:
                        continue
                    Jij = 2.0 * self.k \
                          * (g.adj(r) * field[[self.coord[i]]]).real \
                          * (g.adj(r) * field[[self.coord[j]]]).real
                    p = 1.0 - np.exp(-2.0 * Jij)
                    u = self.rng.uniform_real(min=0.0, max=1.0)
                    if u < p:
                        F_new.append(j)
                        C.append(j)
                        if self.verbose:
                            g.message(f"Wolff: Adding site {j} to cluster")
            F = F_new

            nc = len(C)
            if nc > max_len:
                g.message(f"Warning: Cluster size overflow encountered!")
                g.message(f"      cluster size = {nc}")
                break

        t1 = g.time()

        if self.verbose:
            g.message(
                f"Wolff: Create cluster in {t1-t0:.4f} s; cluster size = {nc}"
            )

        t0 = g.time()
        field_new = g.lattice(field)
        field_new @= g(field - 2.0 * g((r * g.adj(r)) * field))

        s = sum([field[[self.coord[i]]][0].real for i in C])
        s_new = sum([field_new[[self.coord[i]]][0].real for i in C])

        p = np.exp(-self.a * (s - s_new))
        u = self.rng.uniform_real(min=0.0, max=1.0)
        accept = False
        if u < p:
            field @= g.copy(field_new)
            accept = True
        t1 = g.time()

        if self.verbose:
            g.message(f"Wolff: Update field in {t1-t0:.4f} s; accept = {accept}")
         
        return accept, nc
