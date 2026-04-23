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
# OUTPUT FOLDER
# =========================
output_dir = os.path.abspath("wyniczek")
os.makedirs(output_dir, exist_ok=True)
print("📁 Saving to:", output_dir)

# =========================
# PLOT STYLE (BLACK BACKGROUND PNG)
# =========================
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white"
})

# =========================
# GRID DEFINITION
# =========================
nx, ny, nz = 50, 50, 50
dx = 0.5e-9

cx, cy, cz = nx//2, ny//2, nz//2

x = (np.arange(nx) - cx) * dx
y = (np.arange(ny) - cy) * dx
z = (np.arange(nz) - cz) * dx

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# =========================
# PHYSICAL CONSTANTS
# =========================
e = 1.602e-19
hbar = 1.0545718e-34
me = 9.109e-31

# =========================
# GEOMETRY (QUANTUM DOT)
# =========================
def dot_mask(X, Y, Z):
    R = 20 * dx
    Hh = 10 * dx
    return (X**2 / R**2 + Y**2 / R**2 + Z**2 / Hh**2) <= 1

inside = dot_mask(X, Y, Z)
outside = ~inside

# =========================
# MATERIAL PARAMETERS (GaAs / AlAs)
# =========================
m_GaAs = 0.067 * me
m_AlAs = 0.15 * me

V0 = 1.0 * e

eps_GaAs = 12.9
eps_AlAs = 10.0

meff = np.where(inside, m_GaAs, m_AlAs)
V = np.where(inside, 0.0, V0)

# =========================
# LAPLACIAN
# =========================
def build_laplacian(nx, ny, nz):
    N = nx * ny * nz

    def index(i, j, k):
        return i + j*nx + k*nx*ny

    rows, cols, data = [], [], []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = index(i, j, k)

                rows.append(p)
                cols.append(p)
                data.append(-6)

                for di, dj, dk in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        rows.append(p)
                        cols.append(index(ni, nj, nk))
                        data.append(1)

    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))

print("🔧 Building Laplacian...")
L = build_laplacian(nx, ny, nz)

# =========================
# HAMILTONIAN
# =========================
print("⚙️ Building Hamiltonian...")

N = nx * ny * nz
V_flat = V.flatten()
meff_flat = meff.flatten()

coeff = (hbar**2) / (2 * meff_flat * dx**2)

rows, cols, data = [], [], []

idx = lambda i,j,k: i + j*nx + k*nx*ny

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

# =========================
# SOLVE
# =========================
print("⏳ Solving eigenproblem...")
start = time.time()

eigvals, eigvecs = spla.eigsh(H, k=3, which='SA')

print("✔ Done in:", round(time.time() - start, 2), "s")

# =========================
# WAVEFUNCTION
# =========================
def normalize(psi):
    norm = np.sum(np.abs(psi)**2) * dx**3
    return psi / np.sqrt(norm)

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

header_grid = (
    "# GaAs/AlAs QUANTUM DOT FULL DATA FILE\n"
    "# Columns: x y z Re(psi) Im(psi) density V m_eff\n"
)

np.savetxt(os.path.join(output_dir, "grid_data.dat"), grid_data, header=header_grid)

energies_data = np.column_stack([np.arange(len(eigvals)), eigvals])
np.savetxt(os.path.join(output_dir, "energies.dat"), energies_data)

density_data = np.column_stack([X.flatten(), Y.flatten(), Z.flatten(), psi2.flatten()])
np.savetxt(os.path.join(output_dir, "density.dat"), density_data)

# =========================
# METADATA
# =========================
metadata = {
    "material_system": "GaAs/AlAs",
    "nx": nx,
    "ny": ny,
    "nz": nz,
    "dx": dx
}

with open(os.path.join(output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)

# =========================
# VISUALIZATION (BLACK BACKGROUND PNG)
# =========================

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


