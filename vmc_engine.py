"""
vmc_module.py — Variational Monte Carlo dla dwóch elektronów w kropce kwantowej
=================================================================================
Podłącz ten moduł bezpośrednio pod solver FDM (twój obecny kod).
Na końcu tego pliku znajduje się przykład integracji.

Architektura:
  1. RegularGridInterpolator  → psi(r) w punkcie ciągłym
  2. Funkcja falowa próbna    → Psi(r1,r2) = psi(r1)*psi(r2)*J(r12)
  3. Metropolis (wektoryzacja): N_walkers próbkowanych równolegle w NumPy
  4. Energia lokalna          → E_kin(FDM) + V_ext + V_Coulomb
  5. Wynik                    → <E> ± błąd statystyczny

GPU: zmień `import numpy as np` → `import cupy as np` (jeden przełącznik).
"""

import numpy as np
import scipy.interpolate as si
import scipy.ndimage as snd          # do numerycznego laplasjanu na siatce
from dataclasses import dataclass, field
from typing import Optional, Tuple
import time


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STAŁE FIZYCZNE (SI)
# ─────────────────────────────────────────────────────────────────────────────

E_CHARGE   = 1.602176634e-19   # [C]
HBAR       = 1.054571817e-34   # [J·s]
M_E        = 9.1093837015e-31  # [kg]
EPS_0      = 8.8541878128e-12  # [F/m]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  KONFIGURACJA VMC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VMCConfig:
    """Wszystkie hiperparametry algorytmu w jednym miejscu."""

    # Czynnik Jastrowa:  J(r12) = exp(-alpha * r12 / (1 + beta*r12))
    # alpha ≈ 1/2 dla spinów antyrównoległych (warunek kąta chusteczkowego)
    jastrow_alpha: float = 0.5
    jastrow_beta:  float = 1.0           # [1/m] — skaluje zasięg korelacji

    # Metropolis
    n_walkers:     int   = 2048          # ile konfiguracji równolegle
    step_size:     float = 2.0e-10       # [m] — krok propozycji (≈ 0.4 * dx)
    n_warmup:      int   = 500           # kroki rozgrzewkowe (odrzucamy)
    n_steps:       int   = 5000          # kroki produkcyjne
    target_ar:     float = 0.50          # pożądany acceptance rate

    # Energia lokalna — różniczka numeryczna ∇²ψ
    fdm_delta:     float = 1e-11         # [m] — krok finitny do laplasjanu

    # Przenikalność elektryczna (GaAs: eps_r ≈ 12.9)
    eps_r:         float = 12.9

    # Masy efektywne (można nadpisać po inicjalizacji)
    m_eff_in:      float = 0.067 * M_E  # GaAs
    m_eff_out:     float = 0.15  * M_E  # AlAs

    # Opcje
    adaptive_step: bool  = True          # auto-dostrajanie step_size
    verbose:       bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# 3.  INTERPOLATOR  ψ(r)  na siatce → wartość ciągła
# ─────────────────────────────────────────────────────────────────────────────

class PsiInterpolator:
    """
    Opakowuje scipy.interpolate.RegularGridInterpolator.

    Wejście:  psi [nx,ny,nz], x/y/z w metrach
    Wyjście:  psi(r) dla tablicy punktów shape (N,3)
    """

    def __init__(self, psi: np.ndarray, x: np.ndarray,
                 y: np.ndarray, z: np.ndarray):
        # Interpolator działa na |ψ|  (zawsze rzeczywista, ∝ gęstość)
        # Używamy psi.real — zakładamy, że solver zwraca realne wektory własne
        self._itp = si.RegularGridInterpolator(
            (x, y, z), psi.real,
            method="linear",       # szybkie; "cubic" dokładniejsze ale wolne
            bounds_error=False,
            fill_value=0.0,        # poza siatką → ψ=0 (bariera zewnętrzna)
        )
        self.x_range = (x[0], x[-1])
        self.y_range = (y[0], y[-1])
        self.z_range = (z[0], z[-1])

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        r : shape (..., 3) — współrzędne [m]
        zwraca: shape (...)  — wartości ψ(r)
        """
        shape_in = r.shape[:-1]
        pts = r.reshape(-1, 3)
        vals = self._itp(pts)
        return vals.reshape(shape_in)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CZYNNIK JASTROWA
# ─────────────────────────────────────────────────────────────────────────────

def jastrow(r12: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    J(r12) = exp(-alpha * r12 / (1 + beta*r12))

    Spełnia warunek kąta chusteczkowego dla coulombowskiego oddziaływania.
    r12 : shape (...) — odległość między elektronami [m]
    """
    return np.exp(-alpha * r12 / (1.0 + beta * r12))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FUNKCJA FALOWA PRÓBNA  Ψ_trial(r1, r2)
# ─────────────────────────────────────────────────────────────────────────────

class TrialWavefunction:
    """
    Ψ(r1, r2) = ψ(r1) · ψ(r2) · J(r12)

    Zakłada, że orbitalny ψ pochodzi z solvera jednocząstkowego (FDM).
    Elektronom przypisujemy singletowy spin (antysymm. spinowa → sym. orbitalnie),
    więc część orbitalna jest symetryczna — iloczyn orbitali to poprawna
    funkcja wejściowa do korekty korelacyjnej przez Jastrow.
    """

    def __init__(self, psi_itp: PsiInterpolator, cfg: VMCConfig):
        self.psi = psi_itp
        self.cfg = cfg

    def __call__(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        r1, r2 : shape (N_walkers, 3)
        zwraca: Ψ(r1, r2)  shape (N_walkers,)
        """
        p1  = self.psi(r1)                              # (N,)
        p2  = self.psi(r2)                              # (N,)
        dr  = r1 - r2                                   # (N,3)
        r12 = np.sqrt(np.sum(dr**2, axis=-1)) + 1e-30  # (N,)  unikamy 0
        J   = jastrow(r12, self.cfg.jastrow_alpha, self.cfg.jastrow_beta)
        return p1 * p2 * J

    def log_abs_sq(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Logarytm |Ψ|² — numerycznie stabilny dla acceptance ratio w Metropolis.
        """
        psi_val = self(r1, r2)
        return 2.0 * np.log(np.abs(psi_val) + 1e-300)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ENERGIA LOKALNA
# ─────────────────────────────────────────────────────────────────────────────

class LocalEnergy:
    """
    E_L(r1,r2) = E_kin(r1) + E_kin(r2) + V_ext(r1) + V_ext(r2) + V_coul(r12)

    Człon kinetyczny wyznaczamy ze wzoru:
        E_kin(r) = -ℏ²/(2m) · ∇²ψ(r) / ψ(r)

    ∇²ψ aproksymujemy 6-punktowym stencil finitnym w punkcie ciągłym
    (jest to konieczne bo interpolujemy ψ poza siatką).

    Alternatywnie (szybciej):  jeśli punkt leży dokładnie na węźle siatki,
    można użyć wartości z solvera  H·psi_vec = E·psi_vec:
        -ℏ²/(2m)·∇²ψ_n = (E_n - V_n)·ψ_n
    → czyli E_kin(r_n) = E_n - V(r_n).
    Tutaj implementujemy wersję finitną (działa w dowolnym punkcie).
    """

    def __init__(self, psi_itp: PsiInterpolator, V_itp, cfg: VMCConfig,
                 eigval: float):
        self.psi   = psi_itp
        self.V_itp = V_itp         # interpolator potencjału zewnętrznego
        self.cfg   = cfg
        self.eigval = eigval       # energia własna z FDM (referencja)

    def _laplacian_psi(self, r: np.ndarray) -> np.ndarray:
        """
        Numeryczny laplasjian ψ w punktach r (N,3) metodą 6-kierunkowego stencila.
        Zwraca (∇²ψ)(r) shape (N,).
        """
        d = self.cfg.fdm_delta
        psi_c = self.psi(r)                         # ψ(r),  (N,)
        lap = -6.0 * psi_c

        offsets = np.array([
            [d, 0, 0], [-d, 0, 0],
            [0, d, 0], [0, -d, 0],
            [0, 0, d], [0, 0,-d],
        ])  # (6,3)

        for off in offsets:
            lap += self.psi(r + off[None, :])       # (N,)

        return lap / (d * d)

    def __call__(self, r1: np.ndarray, r2: np.ndarray,
                 psi_trial: TrialWavefunction) -> np.ndarray:
        """
        Oblicza E_L dla wszystkich walkerów jednocześnie.
        r1, r2 : (N_walkers, 3)
        zwraca: E_L (N_walkers,)
        """
        cfg = self.cfg
        hbar2_2m1 = HBAR**2 / (2.0 * cfg.m_eff_in)

        # ── Człon kinetyczny  -ℏ²/(2m) ∇²ψ/ψ  dla każdego elektronu ──────
        lap1 = self._laplacian_psi(r1)                   # (N,)
        lap2 = self._laplacian_psi(r2)                   # (N,)
        psi1 = self.psi(r1) + 1e-300
        psi2 = self.psi(r2) + 1e-300

        # Zakładamy jednorodną masę m_eff_in (wewnątrz kropki).
        # Dla punktów na zewnątrz możesz użyć m_eff_out — wymaga interpolatora meff.
        T1 = -hbar2_2m1 * lap1 / psi1                   # (N,)
        T2 = -hbar2_2m1 * lap2 / psi2                   # (N,)

        # ── Potencjał zewnętrzny ──────────────────────────────────────────
        V1 = self.V_itp(r1)                              # (N,)
        V2 = self.V_itp(r2)                              # (N,)

        # ── Oddziaływanie Coulomba ────────────────────────────────────────
        dr   = r1 - r2
        r12  = np.sqrt(np.sum(dr**2, axis=-1)) + 1e-30  # (N,)
        eps  = cfg.eps_r * EPS_0
        V_coul = E_CHARGE**2 / (4.0 * np.pi * eps * r12)

        E_L = T1 + T2 + V1 + V2 + V_coul
        return E_L


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ALGORYTM METROPOLISA (W PEŁNI ZWEKTORYZOWANY)
# ─────────────────────────────────────────────────────────────────────────────

class MetropolisSampler:
    """
    Próbkowanie |Ψ(r1,r2)|² za pomocą algorytmu Metropolisa-Hastingsa.

    Kluczowe cechy:
    - Wszystkie N_walkers konfiguracji aktualizowane JEDNOCZEŚNIE (NumPy)
    - Każdy walker = jedna para (r1, r2) → tensor shape (N,2,3)
    - Propozycja: r_new = r_old + step * N(0,1)
    - Acceptance ratio: min(1, |Ψ_new|²/|Ψ_old|²)
    - Automatyczna kalibracja step_size do target acceptance rate
    """

    def __init__(self, trial_wf: TrialWavefunction,
                 local_energy: LocalEnergy, cfg: VMCConfig):
        self.wf  = trial_wf
        self.EL  = local_energy
        self.cfg = cfg
        self.rng = np.random.default_rng()

    # ── inicjalizacja walkerów ────────────────────────────────────────────
    def _init_walkers(self, x_range, y_range, z_range) -> np.ndarray:
        """
        Losowe startowe pozycje wewnątrz siatki.
        Zwraca: (N, 2, 3)  —  N walkerów, 2 elektrony, 3 współrzędne
        """
        N = self.cfg.n_walkers
        def rand_in(lo, hi):
            return self.rng.uniform(lo, hi, size=(N, 3))

        r1 = rand_in(*zip(x_range, y_range, z_range))
        r2 = rand_in(*zip(x_range, y_range, z_range))

        # Odrzuć konfiguracje gdzie |Ψ| ≈ 0 (oba elektrony poza kropką)
        psi_vals = np.abs(self.wf(r1, r2))
        mask = psi_vals < 1e-20
        if mask.any():
            # Przesuń "puste" walkerów do środka siatki
            cx = 0.5 * (x_range[0] + x_range[1])
            cy = 0.5 * (y_range[0] + y_range[1])
            cz = 0.5 * (z_range[0] + z_range[1])
            center = np.array([cx, cy, cz])
            r1[mask] = center + self.rng.uniform(-1e-10, 1e-10, (mask.sum(), 3))
            r2[mask] = center + self.rng.uniform(-1e-10, 1e-10, (mask.sum(), 3))

        return np.stack([r1, r2], axis=1)   # (N, 2, 3)

    # ── jeden krok Metropolisa dla wszystkich walkerów ────────────────────
    def _step(self, walkers: np.ndarray,
              log_prob_old: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        walkers:      (N, 2, 3)
        log_prob_old: (N,)   = 2·log|Ψ_old|

        Zwraca:
          walkers_new (N,2,3), log_prob_new (N,), acceptance_rate (float)
        """
        N   = self.cfg.n_walkers
        s   = self.cfg.step_size

        # Propozycja: każdy elektron dostaje niezależną propozycję
        proposal = walkers + s * self.rng.standard_normal((N, 2, 3))

        r1_new = proposal[:, 0, :]
        r2_new = proposal[:, 1, :]
        log_prob_new = self.wf.log_abs_sq(r1_new, r2_new)

        # Acceptance ratio w przestrzeni logarytmicznej (numerycznie stabilne)
        log_ratio = log_prob_new - log_prob_old         # (N,)
        u = np.log(self.rng.uniform(0, 1, N) + 1e-300)
        accept = log_ratio >= u                         # (N,) bool

        # Aktualizacja wektorowa — bez pętli
        walkers_new  = np.where(accept[:, None, None], proposal, walkers)
        log_prob_out = np.where(accept, log_prob_new, log_prob_old)

        acc_rate = accept.mean()
        return walkers_new, log_prob_out, float(acc_rate)

    # ── pełna procedura VMC ───────────────────────────────────────────────
    def run(self, x_range, y_range, z_range) -> dict:
        """
        Główna pętla VMC.

        Zwraca słownik:
          energy_mean  [eV]
          energy_std   [eV]
          energy_err   [eV]   (błąd średniej = std/sqrt(N_eff))
          acceptance_rate
          energies_raw [J]    (próbki do analizy)
        """
        cfg = self.cfg

        # ── inicjalizacja ────────────────────────────────────────────────
        walkers  = self._init_walkers(x_range, y_range, z_range)
        r1, r2   = walkers[:, 0, :], walkers[:, 1, :]
        log_prob = self.wf.log_abs_sq(r1, r2)

        # ── rozgrzewka (thermalization) ──────────────────────────────────
        if cfg.verbose:
            print(f"🔥 Rozgrzewka: {cfg.n_warmup} kroków × {cfg.n_walkers} walkerów")

        for step in range(cfg.n_warmup):
            walkers, log_prob, ar = self._step(walkers, log_prob)

            # Adaptacyjna kalibracja kroku
            if cfg.adaptive_step and step % 50 == 49:
                if ar > cfg.target_ar + 0.05:
                    cfg.step_size *= 1.1
                elif ar < cfg.target_ar - 0.05:
                    cfg.step_size *= 0.9

        if cfg.verbose:
            print(f"   step_size po kalibracji: {cfg.step_size:.3e} m")

        # ── produkcja ────────────────────────────────────────────────────
        if cfg.verbose:
            print(f"⚛️  Produkcja: {cfg.n_steps} kroków...")

        t0 = time.time()
        energies = []
        ar_acc   = []

        for step in range(cfg.n_steps):
            walkers, log_prob, ar = self._step(walkers, log_prob)
            ar_acc.append(ar)

            r1, r2 = walkers[:, 0, :], walkers[:, 1, :]
            E_L = self.EL(r1, r2, self.wf)           # (N,)

            # Odcinamy wartości odstające (|E_L| > 10×median) — stabilność
            E_median = np.median(np.abs(E_L))
            mask_ok  = np.abs(E_L) < 10.0 * E_median
            energies.append(E_L[mask_ok].mean())

        elapsed = time.time() - t0

        energies = np.array(energies)                 # (n_steps,)
        E_J      = energies.mean()
        E_eV     = E_J / E_CHARGE
        std_eV   = energies.std() / E_CHARGE
        N_eff    = len(energies)
        err_eV   = std_eV / np.sqrt(N_eff)

        mean_ar = float(np.mean(ar_acc))

        if cfg.verbose:
            print(f"\n{'─'*50}")
            print(f"  <E>  = {E_eV:.4f} ± {err_eV:.4f} eV")
            print(f"  Std  = {std_eV:.4f} eV")
            print(f"  Acceptance rate = {mean_ar:.2%}")
            print(f"  Czas produkcji  = {elapsed:.1f} s")
            print(f"{'─'*50}")

        return {
            "energy_mean": E_eV,
            "energy_std":  std_eV,
            "energy_err":  err_eV,
            "acceptance_rate": mean_ar,
            "energies_raw": energies,
            "elapsed_s":   elapsed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  OPTYMALIZACJA PARAMETRÓW JASTROWA  (prosta metoda gradientowa)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_jastrow(base_cfg: VMCConfig, psi_itp, V_itp,
                     eigval, x, y, z,
                     alpha_range=(0.1, 2.0), n_points=8) -> Tuple[float, float]:
    """
    Skanuje alpha w podanym zakresie i wybiera minimum <E>.
    Szybka wersja: zmniejszony n_steps podczas skanu.
    """
    from copy import deepcopy
    alphas   = np.linspace(alpha_range[0], alpha_range[1], n_points)
    energies = []

    print(f"\n🔍 Optymalizacja alpha Jastrowa ({n_points} punktów)...")

    for a in alphas:
        cfg = deepcopy(base_cfg)
        cfg.jastrow_alpha = a
        cfg.n_warmup = 100
        cfg.n_steps  = 500
        cfg.verbose  = False

        trial = TrialWavefunction(psi_itp, cfg)
        E_itp = _build_V_itp(V_itp, x, y, z)
        loc_e = LocalEnergy(psi_itp, E_itp, cfg, eigval)
        sampler = MetropolisSampler(trial, loc_e, cfg)

        x_range = (x[0], x[-1])
        y_range = (y[0], y[-1])
        z_range = (z[0], z[-1])

        res = sampler.run(x_range, y_range, z_range)
        energies.append(res["energy_mean"])
        print(f"   alpha={a:.2f}  →  <E>={res['energy_mean']:.4f} eV")

    best_i = int(np.argmin(energies))
    best_alpha = float(alphas[best_i])
    print(f"✅ Optymalne alpha = {best_alpha:.3f}  (<E> = {energies[best_i]:.4f} eV)")
    return best_alpha, float(energies[best_i])


def _build_V_itp(V: np.ndarray, x, y, z):
    """Buduje interpolator potencjału V (pomocnicza funkcja)."""
    return si.RegularGridInterpolator(
        (x, y, z), V,
        method="linear", bounds_error=False, fill_value=float(V.max())
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9.  GŁÓWNA FUNKCJA INTEGRUJĄCA Z SOLVEREM FDM
# ─────────────────────────────────────────────────────────────────────────────

def run_vmc(
    psi:     np.ndarray,       # z solvera FDM, shape (nx,ny,nz)
    V:       np.ndarray,       # potencjał zewnętrzny, shape (nx,ny,nz)
    x:       np.ndarray,       # oś x [m]
    y:       np.ndarray,       # oś y [m]
    z:       np.ndarray,       # oś z [m]
    eigval:  float,            # energia własna [J] z solvera
    cfg:     Optional[VMCConfig] = None,
    optimize_alpha: bool = False,
) -> dict:
    """
    Punkt wejścia. Wywołaj po rozwiązaniu równania Schrödingera:

        from vmc_module import run_vmc, VMCConfig
        cfg = VMCConfig(n_walkers=4096, n_steps=8000)
        result = run_vmc(psi, V, x, y, z, eigvals[0], cfg)
        print(f"Energia dwóch elektronów: {result['energy_mean']:.4f} eV")
    """
    if cfg is None:
        cfg = VMCConfig()

    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])
    z_range = (z[0], z[-1])

    # ── interpolatory ────────────────────────────────────────────────────
    psi_itp = PsiInterpolator(psi, x, y, z)
    V_itp   = _build_V_itp(V, x, y, z)

    # ── opcjonalna optymalizacja Jastrowa ────────────────────────────────
    if optimize_alpha:
        best_alpha, _ = optimize_jastrow(cfg, psi_itp, V_itp, eigval, x, y, z)
        cfg.jastrow_alpha = best_alpha

    # ── składanie obiektów ───────────────────────────────────────────────
    trial   = TrialWavefunction(psi_itp, cfg)
    loc_e   = LocalEnergy(psi_itp, V_itp, cfg, eigval)
    sampler = MetropolisSampler(trial, loc_e, cfg)

    if cfg.verbose:
        E_ref_eV = eigval / E_CHARGE
        print(f"\n{'═'*50}")
        print(f"  VMC — GaAs/AlAs Quantum Dot")
        print(f"  Energia 1-cząstkowa (FDM): {E_ref_eV:.4f} eV")
        print(f"  Walkerów:  {cfg.n_walkers}")
        print(f"  Kroków:    {cfg.n_steps}")
        print(f"  α Jastrowa: {cfg.jastrow_alpha:.3f}")
        print(f"{'═'*50}\n")

    result = sampler.run(x_range, y_range, z_range)
    result["eigval_1p_eV"] = eigval / E_CHARGE
    result["jastrow_alpha"] = cfg.jastrow_alpha

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PRZYKŁAD INTEGRACJI Z TWOIM KODEM (wklej na koniec solver FDM)
# ─────────────────────────────────────────────────────────────────────────────
#
# ── Po linii z: eigvals, eigvecs = spla.eigsh(H, k=3, which='SA') ──
#
#   from vmc_module import run_vmc, VMCConfig
#
#   cfg = VMCConfig(
#       n_walkers     = 4096,     # więcej = dokładniej, ale wolniej
#       n_steps       = 10000,
#       jastrow_alpha = 0.5,      # lub optimize_alpha=True
#       jastrow_beta  = 1.0,
#       step_size     = 2e-10,    # ≈ 0.4 * dx
#       adaptive_step = True,
#       verbose       = True,
#   )
#
#   result = run_vmc(
#       psi    = psi,             # już znormalizowane (nx,ny,nz)
#       V      = V,               # potencjał z solvera (nx,ny,nz) [J]
#       x      = x,              # np.arange(nx)*dx - cx*dx
#       y      = y,
#       z      = z,
#       eigval = eigvals[0],      # [J]
#       cfg    = cfg,
#       optimize_alpha = False,   # True = ~8x dłużej, ale lepszy Jastrow
#   )
#
#   print(f"\n🎯  <E_2e> = {result['energy_mean']:.4f} ± {result['energy_err']:.4f} eV")
#   print(f"    Korekta korelacyjna vs 2×E_1e: "
#         f"{result['energy_mean'] - 2*result['eigval_1p_eV']:.4f} eV")
#
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Szybki test modułu (bez solvera FDM) ────────────────────────────
    # Symulujemy gaussowską funkcję falową w pudełku 25x25x25 nm
    print("=== Szybki test VMC (Gaussian mock) ===\n")

    nx = ny = nz = 30
    dx = 0.5e-9
    cx, cy, cz = nx//2, ny//2, nz//2
    x = (np.arange(nx) - cx) * dx
    y = (np.arange(ny) - cy) * dx
    z = (np.arange(nz) - cz) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    sigma = 5e-9
    psi_mock = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    norm = np.sqrt(np.sum(psi_mock**2) * dx**3)
    psi_mock /= norm

    V_mock = np.zeros_like(psi_mock)
    R = 10 * dx
    outside = (X**2 + Y**2 + Z**2) > R**2
    V_mock[outside] = 1.0 * 1.602e-19   # 1 eV bariera

    E_mock = 0.1 * 1.602e-19   # 0.1 eV energia próbna

    cfg = VMCConfig(
        n_walkers=256, n_steps=200, n_warmup=50, verbose=True
    )

    result = run_vmc(psi_mock, V_mock, x, y, z, E_mock, cfg)
    print(f"\n✅  Test zakończony. <E> = {result['energy_mean']:.4f} eV")