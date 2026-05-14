import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from Ci_engine import run_ci, CIConfig # type: ignore


# =========================
# MAIN
# =========================
def main():
    print("🚀 Uruchamiam system FDM + CI (Configuration Interaction)")

    # 1. Solver jednocząstkowy — zwraca teraz też eigvecs (potrzebne do CI)
    eigvals, eigvecs, V, x, y, z, nx, ny, nz = Nanostructure()

    # 2. Konfiguracja CI
    cfg = CIConfig(
        n_orbitals   = 10,        # ile stanów jednocząstkowych w bazie
        n_mc_samples = 100_000,   # próbek MC na całkę Coulomba (więcej = dokładniej)
        spin_state   = 'singlet', # stan podstawowy 2e to singlet
        verbose      = True,
    )

    # 3. Obliczenia CI z całkami Coulomba liczonymi MC
    result = run_ci(eigvals, eigvecs, x, y, z, nx, ny, nz, cfg)

    # 4. Wynik główny
    print(f"\n🎯 FINALNY WYNIK:")
    print(f"   E_1  = {result['E_1_eV']:.4f} eV   (jednocząstkowe, z FDM)")
    print(f"   E_2  = {result['E_2_eV']:.4f} eV   (dwucząstkowe, CI+MC)")
    print(f"   E_X  = {result['E_X_eV']:+.4f} eV   (energia korelacji = E_2 - 2·E_1)")


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
        R  = 20 * dx
        Hh = 10 * dx
        return (X**2 / R**2 + Y**2 / R**2 + Z**2 / Hh**2) <= 1

    inside = dot_mask(X, Y, Z)

    # MATERIAL
    m_GaAs = 0.067 * me
    m_AlAs = 0.15  * me
    V0     = 1.0 * e

    meff = np.where(inside, m_GaAs, m_AlAs)
    V    = np.where(inside, 0.0, V0)

    # INDEX
    idx = lambda i, j, k: i + j*nx + k*nx*ny

    # HAMILTONIAN (poprawiony znak: człon kinetyczny ma +6 na diagonali)
    N_grid  = nx * ny * nz
    V_flat  = V.flatten()
    meff_flat = meff.flatten()

    coeff = (hbar**2) / (2 * meff_flat * dx**2)

    rows, cols, data = [], [], []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p  = idx(i, j, k)
                cp = coeff[p]

                # Człon diagonalny: +6·coeff (z laplasjanu FDM)
                rows.append(p); cols.append(p); data.append(+6 * cp)

                # Sąsiedzi: -coeff
                for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        q = idx(ni, nj, nk)
                        rows.append(p); cols.append(q); data.append(-cp)

    H_sparse = sp.csr_matrix((data, (rows, cols)), shape=(N_grid, N_grid))
    H_sparse = H_sparse + sp.diags(V_flat)

    # SOLVE — potrzebujemy więcej wektorów własnych dla bazy CI
    n_eig = 10   # tyle ile n_orbitals w CIConfig
    print(f"⏳ Solving eigenproblem (k={n_eig})...")
    eigvals, eigvecs = spla.eigsh(H_sparse, k=n_eig, which='SA')

    # Sortuj rosnąco (eigsh nie gwarantuje kolejności)
    order   = np.argsort(eigvals)
    eigvals = eigvals[order]      # [J]
    eigvecs = eigvecs[:, order]   # shape (N_grid, n_eig)

    # NORMALIZE każdy wektor osobno
    for k in range(eigvecs.shape[1]):
        norm =   dx**3 #usunąć 
        eigvecs[:, k] /= norm

    # SAVE
    np.savetxt(os.path.join(output_dir, "energies.dat"), eigvals)

    e_charge = 1.602e-19
    print(f"\nEnergie jednocząstkowe (pierwsze {n_eig}):")
    for i, ev in enumerate(eigvals):
        print(f"  ε_{i} = {ev/e_charge:.4f} eV")
    # =========================
    # WAVEFUNCTION
    # =========================
    def normalize(psi):
        norm = np.sum(np.abs(psi)**2) * dx**3
        return psi / np.sqrt(norm)

    # =========================
    # VISUALIZATION (BLACK BACKGROUND PNG)
    # =========================

    psi = eigvecs[:, 0].reshape((nx, ny, nz))
    psi = normalize(psi)
    psi2 = np.abs(psi)**2

    # =========================
    # SAVE .DAT FILES WITH FULL HEADERS
    # =========================
    print("💾 Saving .dat files...")

    grid_data = np.column_stack([
        X.flatten(), Y.flatten(), Z.flatten(),
        psi.real.flatten(), psi.imag.flatten(),
        psi2.flatten(), V.flatten(), meff.flatten()
    ])

    def save(fig, name):
        fig.savefig(os.path.join(output_dir, name), dpi=300, bbox_inches="tight", facecolor='black')
        plt.close(fig)

    fig = plt.figure(facecolor='black')
    plt.imshow(psi2[:, :, cz].T, origin="lower", cmap="Oranges")
    plt.colorbar()
    plt.title("GaAs/AlAs density XY")
    save(fig, "xy.png")

    fig = plt.figure(facecolor='black')
    plt.imshow(psi2[:, cy, :].T, origin="lower", cmap="Oranges")
    plt.colorbar()
    plt.title("GaAs/AlAs density XZ")
    save(fig, "xz.png")

    fig = plt.figure(facecolor='black')
    proj = np.sum(psi2, axis=2)
    plt.imshow(proj.T, origin="lower", cmap="Oranges")
    plt.colorbar()
    plt.title("Projection Z")
    save(fig, "proj.png")

    return eigvals, eigvecs, V, x, y, z, nx, ny, nz


# =========================
# COULOMB (nieużywane — zastąpione przez ci_engine)
# =========================
def Coulomb(Psi):
    print("⚡ Coulomb jest teraz w ci_engine.py (CoulombIntegralsMC)")
    return None


# =========================
# MANY BODY (nieużywane — zastąpione przez ci_engine)
# =========================
def ManyBody(basis_size=0, single_eigval=0, single_eigvec=0, coul_matrix=0):
    """
    Logika przeniesiona do ci_engine.py:
      - CIBasis    — buduje bazę (odpowiednik pętli generującej basis[])
      - build_hamiltonian — buduje macierz H
      - solve_ci   — diagonalizuje i liczy E_X
    """
    print("ℹ️  ManyBody zastąpione przez ci_engine.run_ci()")
    return


# =========================
if __name__ == "__main__":
    main()