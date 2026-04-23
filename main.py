
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json
import time

# =========================
# MAIN
# =========================
def main():
    print("AAAAAAAAAAAAAAAAA")

    energies, psi = Nanostructure()
    Vmn = Coulomb(psi)
    ManyBody(energies, Vmn)
    return


# =========================
# NANOSTRUCTURE
# =========================
def Nanostructure():

    output_dir = os.path.abspath("wyniczek")
    os.makedirs(output_dir, exist_ok=True)

    # GRID
    nx, ny, nz = 50, 50, 50
    dx = 0.5e-9

    cx, cy, cz = nx//2, ny//2, nz//2

    x = (np.arange(nx) - cx) * dx
    y = (np.arange(ny) - cy) * dx
    z = (np.arange(nz) - cz) * dx

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # CONST
    e = 1.602e-19
    hbar = 1.0545718e-34
    me = 9.109e-31

    # GEOMETRY
    def dot_mask(X, Y, Z):
        R = 20 * dx
        Hh = 10 * dx
        return (X**2 / R**2 + Y**2 / R**2 + Z**2 / Hh**2) <= 1

    inside = dot_mask(X, Y, Z)

    # MATERIAL
    m_GaAs = 0.067 * me
    m_AlAs = 0.15 * me
    V0 = 1.0 * e

    meff = np.where(inside, m_GaAs, m_AlAs)
    V = np.where(inside, 0.0, V0)

    # INDEX
    idx = lambda i,j,k: i + j*nx + k*nx*ny

    # HAMILTONIAN
    N = nx * ny * nz
    V_flat = V.flatten()
    meff_flat = meff.flatten()

    coeff = (hbar**2) / (2 * meff_flat * dx**2)

    rows, cols, data = [], [], []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = idx(i,j,k)

                rows.append(p)
                cols.append(p)
                data.append(-6 * coeff[p])

                cp = coeff[p]

                for di,dj,dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        q = idx(ni,nj,nk)
                        rows.append(p)
                        cols.append(q)
                        data.append(cp)

    H = sp.csr_matrix((data, (rows, cols)), shape=(N, N)) + sp.diags(V_flat)

    # SOLVE
    print("⏳ Solving eigenproblem...")
    eigvals, eigvecs = spla.eigsh(H, k=3, which='SA')

    # NORMALIZE
    def normalize(psi):
        norm = np.sum(np.abs(psi)**2) * dx**3
        return psi / np.sqrt(norm)

    psi = eigvecs[:, 0].reshape((nx, ny, nz))
    psi = normalize(psi)

    # SAVE (opcjonalnie)
    np.savetxt(os.path.join(output_dir, "energies.dat"), eigvals)

    return eigvals, psi


# =========================
# COULOMB
# =========================
def Coulomb(Psi):
    print("⚡ Coulomb not implemented yet")
    return None


# =========================
# MANY BODY
# =========================

def ManyBody(basis_size=0, single_eigval=0, single_eigvec=0, coul_matrix=0):
    #Znajduje energie układu wielocząstkowego
    basis_cutoff = 10 

    basis = []

    for i in range(1, basis_cutoff):
        for j in range(i, basis_cutoff):
            basis.append([i, j, 1, 2])
            
            if i != j:
                basis.append([i, j, 2, 1])
                basis.append([i, j, 1, 1])
                basis.append([i, j, 2, 2])

    basis = np.array(basis)
    print(basis)
    return

# =========================
if __name__ == "__main__":
    main()
