"""
vmc_engine.py — Variational Monte Carlo dla dwóch elektronów w kropce kwantowej GaAs/AlAs
===========================================================================================

MATEMATYCZNE PODSTAWY 
---------------------------------------

Szukamy wartości oczekiwanej energii układu dwóch elektronów:

    <E> = <Ψ|Ĥ|Ψ> / <Ψ|Ψ>

gdzie Hamiltonian dwuciałowy to:

    Ĥ = T̂₁ + T̂₂ + V_ext(r₁) + V_ext(r₂) + V_C(r₁₂)

    T̂ᵢ = -ℏ²/(2m) ∇²ᵢ           (energia kinetyczna i-tego elektronu)
    V_ext(r)  = potencjał studni kwantowej (0 wewnątrz, V₀ na zewnątrz)
    V_C(r₁₂) = e²/(4πε·r₁₂)     (odpychanie Coulomba, r₁₂ = |r₁ - r₂|)


KLUCZOWY TRIK VMC: "Energia Lokalna"
--------------------------------------

Zamiast liczyć całkę 6D bezpośrednio, zapisujemy:

    <E> = ∫ |Ψ(r₁,r₂)|² · E_L(r₁,r₂) dr₁dr₂
           ─────────────────────────────────────
                  ∫ |Ψ(r₁,r₂)|² dr₁dr₂

gdzie ENERGIA LOKALNA to:

    E_L(r₁,r₂) = Ĥ·Ψ(r₁,r₂) / Ψ(r₁,r₂)

Teraz <E> = E[E_L] = wartość oczekiwana E_L pod miarą |Ψ|²/⟨Ψ|Ψ⟩.

To jest całka 6D, ale szacujemy ją metodą Monte Carlo:
    → próbkujemy (r₁,r₂) z rozkładu |Ψ|² (algorytm Metropolisa)
    → liczymy średnią arytmetyczną E_L po próbkach

Błąd statystyczny spada jak 1/√N — nie jak 2^(-6N) jak w metodach siatki.


FUNKCJA PRÓBNA (Trial Wavefunction)
-------------------------------------

Wybieramy postać:

    Ψ(r₁,r₂) = φ(r₁) · φ(r₂) · J(r₁₂)

gdzie:
  φ(r)   = jednocząstkowa funkcja falowa z solvera FDM (interpolowana z siatki)
  J(r₁₂) = czynnik Jastrowa — modeluje korelację elektronową

Czynnik Jastrowa (forma Padé, spełnia "cusp condition"):

    J(r₁₂) = exp( r₁₂ / (2·(1 + α·r₁₂)) )

    Uwaga o znaku: wykładnik jest DODATNI przy małych r₁₂
    (elektrony "unikają się nawzajem" — Ψ rośnie gdy są daleko).

    Warunek graniczny (cusp condition dla spinów antyrównoległych):
        dJ/dr₁₂|_{r₁₂→0} = J(0)/2  ✓


PEŁNY LAPLASJIAN FUNKCJI PRÓBNEJ
----------------------------------

To jest najważniejsza poprawka względem wersji naiwnej.
Człon kinetyczny wymaga ∇²Ψ/Ψ, nie ∇²φ/φ.

Niech f = φ(r₁)·φ(r₂), g = J(r₁₂). Wtedy Ψ = f·g.

Laplasjian po r₁:

    ∇²_{r₁} Ψ = (∇²_{r₁} φ₁)·φ₂·g + φ₁·φ₂·∇²_{r₁} g + 2(∇_{r₁} φ₁)·φ₂·(∇_{r₁} g)

Dzieląc przez Ψ = φ₁·φ₂·g:

    ∇²_{r₁} Ψ / Ψ = (∇²φ₁/φ₁) + (∇²_{r₁} g/g) + 2·(∇φ₁/φ₁)·(∇_{r₁} g/g)

Analogicznie dla r₂ (z ∇_{r₂} g = -∇_{r₁} g, bo g zależy od r₁₂ = r₁ - r₂).

Pochodne Jastrowa (obliczane analitycznie — szybciej i dokładniej niż FDM):

    Niech u(r) = r/(2(1+αr)), wtedy J = exp(u).

    du/dr = 1 / (2·(1+αr)²)

    ∇_{r₁} g/g = (du/dr₁₂) · r̂₁₂    gdzie r̂₁₂ = (r₁-r₂)/r₁₂

    ∇²_{r₁} g/g = (d²u/dr²)|_{r₁₂} + (du/dr)|²_{r₁₂} + (2/r₁₂)·(du/dr)|_{r₁₂}
                    (składnik radilanego laplasjanu w 3D)


ALGORYTM METROPOLISA-HASTINGSA
--------------------------------

Próbkuje z |Ψ|²  bez liczenia stałej normalizacji:

    1. Zacznij od konfiguracji (r₁, r₂)
    2. Zaproponuj nową: r₁' = r₁ + δ·η, r₂' = r₂ + δ·η  (η ~ N(0,1))
    3. Oblicz ratio: A = |Ψ(r₁',r₂')|²/|Ψ(r₁,r₂)|²
    4. Akceptuj z prawdopodobieństwem min(1, A)
    5. Zbieraj E_L dla zaakceptowanych (i odrzuconych!) konfiguracji

Wektoryzacja: N_walkers konfiguracji aktualizowanych JEDNOCZEŚNIE przez NumPy.
"""

import numpy as np
import scipy.interpolate as si
from dataclasses import dataclass
from typing import Optional, Tuple
import time


# ─────────────────────────────────────────────────────────────────────────────
# STAŁE FIZYCZNE (SI)
# ─────────────────────────────────────────────────────────────────────────────

E_CHARGE = 1.602176634e-19   # ładunek elektronu [C]
HBAR     = 1.054571817e-34   # stała Plancka/2π [J·s]
M_E      = 9.1093837015e-31  # masa elektronu [kg]
EPS_0    = 8.8541878128e-12  # przenikalność próżni [F/m]


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA VMC
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VMCConfig:
    """
    Wszystkie hiperparametry algorytmu w jednym miejscu.

    Parametr Jastrowa α kontroluje siłę korelacji:
        α → 0  : J → 1  (brak korelacji, wynik = 2·E_1cząstk)
        α → ∞  : silna korelacja, elektrony mocno się unikają
    Optymalną α wyznaczamy minimalizując <E> (zasada wariacyjna:
        <E>[Ψ_trial] ≥ E_dokładne, więc minimum <E> = najlepsze przybliżenie).
    """

    # Czynnik Jastrowa: J(r₁₂) = exp(r₁₂ / (2·(1 + α·r₁₂)))
    jastrow_alpha: float = 1e8        # [1/m] — parametr wariacyjny, OPTYMALIZOWANY

    # Metropolis
    n_walkers:  int   = 2048          # liczba równoległych konfiguracji
    step_size:  float = 2.0e-10       # δ — krok propozycji [m], ≈ 0.4·dx
    n_warmup:   int   = 500           # kroki termalizacji (odrzucane)
    n_steps:    int   = 5000          # kroki produkcyjne (zbieramy E_L)
    target_ar:  float = 0.50          # docelowy acceptance rate (0.4-0.6 = optymalny)

    # Różniczkowanie numeryczne ∇²φ/φ (krok stencila)
    fdm_delta:  float = 1e-11         # [m], powinno być << dx ale >> błąd numeryczny

    # Materiał: GaAs/AlAs
    eps_r:      float = 12.9          # przenikalność względna GaAs (bezwymiarowa)
    m_eff:      float = 0.067 * M_E  # masa efektywna elektronu w GaAs [kg]

    # Opcje
    adaptive_step: bool = True        # czy auto-dostrajać step_size podczas warmup
    verbose:       bool = True


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATOR φ(r) — siatka dyskretna → wartość ciągła
# ─────────────────────────────────────────────────────────────────────────────

class PsiInterpolator:
    """
    Opakowuje scipy.interpolate.RegularGridInterpolator.

    Problem: solver FDM daje φ tylko na węzłach siatki (nx·ny·nz liczb).
    Potrzebujemy φ(r) dla DOWOLNEGO r ∈ ℝ³ (walkery lądują gdziekolwiek).

    Rozwiązanie: interpolacja trójliniowa między węzłami.

    Wejście konstruktora:
        psi : array (nx, ny, nz) — jednocząstkowa f. falowa z FDM
        x, y, z : array 1D — współrzędne siatki [m]

    Wywołanie: psi_itp(r) gdzie r ma shape (..., 3)
    """

    def __init__(self, psi: np.ndarray, x: np.ndarray,
                 y: np.ndarray, z: np.ndarray):
        # Używamy psi.real (eigvecs mogą być zespolone z powodów numerycznych,
        # ale dla potencjału rzeczywistego wybieramy realne reprezentanty)
        self._itp = si.RegularGridInterpolator(
            (x, y, z), psi.real,
            method="cubic",      # linear - trójliniowa — szybka, C¹ wewnątrz komórki cubic
            bounds_error=False,
            fill_value=0.0,       # poza siatką φ = 0 (bariera jest nieskończona)
        )
        self.x_range = (float(x[0]), float(x[-1]))
        self.y_range = (float(y[0]), float(y[-1]))
        self.z_range = (float(z[0]), float(z[-1]))

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        r : shape (..., 3)
        zwraca: φ(r), shape (...)
        """
        shape_in = r.shape[:-1]
        vals = self._itp(r.reshape(-1, 3))
        return vals.reshape(shape_in)


# ─────────────────────────────────────────────────────────────────────────────
# CZYNNIK JASTROWA i jego pochodne (ANALITYCZNE)
# ─────────────────────────────────────────────────────────────────────────────

def jastrow_and_derivs(r12: np.ndarray, alpha: float):
    """
    Oblicza czynnik Jastrowa i jego pochodne analitycznie.

    Forma: J(r) = exp( u(r) ),  u(r) = r / (2·(1 + α·r))

    Pochodna:      du/dr = 1 / (2·(1+αr)²)
    Druga pochodna: d²u/dr² = -α / (1+αr)³

    Argumenty:
        r12   : array (...) — odległość |r₁ - r₂| [m], musi być > 0
        alpha  : parametr wariacyjny [1/m]

    Zwraca tuple:
        J       : exp(u),         shape (...)
        du_dr   : du/dr,          shape (...)   — potrzebne do członu gradientowego
        d2u_dr2 : d²u/dr²,        shape (...)   — potrzebne do członu laplasjanu
    """
    denom  = 1.0 + alpha * r12          # (1 + αr)
    u      = r12 / (2.0 * denom)        # u(r) = r/(2(1+αr))
    J      = np.exp(u)                  # czynnik Jastrowa
    du_dr  = 1.0 / (2.0 * denom**2)    # du/dr
    d2u_dr2 = -alpha / denom**3         # d²u/dr²

    return J, du_dr, d2u_dr2


# ─────────────────────────────────────────────────────────────────────────────
# FUNKCJA PRÓBNA Ψ(r₁, r₂)
# ─────────────────────────────────────────────────────────────────────────────

class TrialWavefunction:
    """
    Ψ(r₁, r₂) = φ(r₁) · φ(r₂) · J(r₁₂)

    Własności:
    - Symetryczna względem zamiany r₁↔r₂ (elektrony w stanie singletowym
      mają antysymetryczny spin → część orbitalna musi być symetryczna)
    - Spełnia cusp condition Kato: dΨ/dr₁₂|_{r₁₂→0} = Ψ(0)/2  ← dzięki J
    - Czynnik Jastrowa modeluje korelację: dwa elektrony "unikają się"
      bardziej niż przewiduje przybliżenie Hartree (φ₁·φ₂)

    Sens fizyczny α:
        Większe α  → silniejsza korelacja → elektrony dalej od siebie
        α = 0      → Ψ = φ₁·φ₂, brak korelacji (niezależne elektrony)
    """

    def __init__(self, psi_itp: PsiInterpolator, cfg: VMCConfig):
        self.psi = psi_itp
        self.cfg = cfg

    def __call__(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        r1, r2 : shape (N, 3)
        zwraca: Ψ(r₁,r₂), shape (N,)
        """
        phi1  = self.psi(r1)
        phi2  = self.psi(r2)
        dr    = r1 - r2
        r12   = np.sqrt(np.sum(dr**2, axis=-1)) + 1e-30   # unikamy 0
        J, _, _ = jastrow_and_derivs(r12, self.cfg.jastrow_alpha)
        return phi1 * phi2 * J

    def log_abs_sq(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Logarytm |Ψ|² — stabilny numerycznie dla acceptance ratio.

        Metropolis wymaga tylko stosunku |Ψ_nowe|²/|Ψ_stare|², który
        w przestrzeni log to różnica. Unikamy przez to błędów underflow
        gdy Ψ ≈ 0 (elektrony blisko węzłów φ).
        """
        psi_val = self(r1, r2)
        return 2.0 * np.log(np.abs(psi_val) + 1e-300)


# ─────────────────────────────────────────────────────────────────────────────
# ENERGIA LOKALNA — serce obliczeń VMC
# ─────────────────────────────────────────────────────────────────────────────

class LocalEnergy:
    """
    E_L(r₁,r₂) = ĤΨ(r₁,r₂) / Ψ(r₁,r₂)

    Rozkładamy Hamiltonian:
        E_L = E_kin,1 + E_kin,2 + V_ext(r₁) + V_ext(r₂) + V_Coulomb(r₁₂)

    ──────────────────────────────────────────────────────────────────────────
    CZŁON KINETYCZNY — WAŻNA POPRAWKA
    ──────────────────────────────────────────────────────────────────────────

    Naiwnie można by myśleć: E_kin,1 = -ℏ²/(2m) · ∇²φ(r₁)/φ(r₁)
    To BYŁOBY BŁĘDEM. Ψ = φ₁·φ₂·J, więc ∇²_{r₁}Ψ ≠ (∇²φ₁)·φ₂·J.

    Poprawna reguła różniczkowania dla Ψ = f·g (f=φ₁φ₂, g=J):

        ∇²_{r₁}Ψ / Ψ  =  ∇²φ₁/φ₁  +  ∇²_{r₁}J/J  +  2·(∇φ₁/φ₁)·(∇_{r₁}J/J)

    Trzy człony:
      [A] ∇²φ₁/φ₁       — laplasjian jednocząstkowy (FDM numerycznie)
      [B] ∇²_{r₁}J/J    — laplasjian Jastrowa po r₁ (ANALITYCZNIE)
      [C] 2·(∇φ₁/φ₁)·(∇_{r₁}J/J)  — człon mieszany gradient×gradient

    ──────────────────────────────────────────────────────────────────────────
    POCHODNE JASTROWA (ANALITYCZNE)
    ──────────────────────────────────────────────────────────────────────────

    J zależy od r₁ tylko przez r₁₂ = |r₁ - r₂|. Oznaczmy r̂ = (r₁-r₂)/r₁₂.

    Gradient:
        ∇_{r₁}J/J = (du/dr)|_{r₁₂} · r̂₁₂

    Laplasjian radialny w 3D:
        ∇²_{r₁}J/J = (d²u/dr²)|_{r₁₂} + (du/dr)²|_{r₁₂} + (2/r₁₂)·(du/dr)|_{r₁₂}

    Skąd 2/r₁₂? — to człon kątowy w laplasjanie sferycznym: ∇² = d²/dr² + (2/r)·d/dr

    ──────────────────────────────────────────────────────────────────────────
    CZŁON COULOMBA — ZAWSZE OBECNY
    ──────────────────────────────────────────────────────────────────────────

        V_C(r₁₂) = e² / (4πε·r₁₂)

    To jest skalarne oddziaływanie między dwoma elektronami — NIE zanika,
    NIE jest przybliżone, jest liczone dokładnie dla każdej konfiguracji.
    """

    def __init__(self, psi_itp: PsiInterpolator, V_itp, cfg: VMCConfig):
        self.psi   = psi_itp
        self.V_itp = V_itp
        self.cfg   = cfg

    # ── Laplasjian ∇²φ/φ (numeryczny, 6-kierunkowy stencil) ──────────────

    def _laplacian_phi_over_phi(self, r: np.ndarray) -> np.ndarray:
        """
        Numeryczne ∇²φ(r)/φ(r) metodą 6-punktowego stencila finitnego.

        Stencil: ∇²f ≈ (f(r+δx̂) + f(r-δx̂) + ... - 6f(r)) / δ²

        Uwaga: używamy δ << dx, ale δ nie może być za małe
        (błąd zaokrąglenia rośnie jak ~ε_mach/δ²).
        Optymalne δ ≈ ε_mach^(1/4) · r_char ≈ 1e-11 m dla r_char ~ 1nm.

        Argumenty:
            r : shape (N, 3) — pozycje walkerów [m]
        Zwraca:
            ∇²φ/φ : shape (N,)
        """
        d   = self.cfg.fdm_delta
        phi = self.psi(r) + 1e-300   # φ(r), unikamy dzielenia przez 0

        lap = -6.0 * phi

        offsets = [
            [d, 0, 0], [-d, 0, 0],
            [0, d, 0], [0, -d, 0],
            [0, 0, d], [0, 0, -d],
        ]
        for off in offsets:
            lap += self.psi(r + np.array(off))   # broadcasting: r ma shape (N,3)

        return lap / (d * d * phi)   # ∇²φ/φ, shape (N,)

    # ── Laplasjian Jastrowa po r₁ (analityczny) ──────────────────────────

    def _jastrow_kinetic_terms(self, r1: np.ndarray, r2: np.ndarray):
        """
        Liczy człony [B] i [C] z rozpisanego ∇²Ψ/Ψ.

        Używamy analitycznych pochodnych J — dokładniej i szybciej niż FDM.

        Zwraca:
            lap_J_over_J : (∇²_{r₁}J)/J + (∇²_{r₂}J)/J, shape (N,)
            cross_term   : 2·[(∇φ₁/φ₁)·(∇_{r₁}J/J) + (∇φ₂/φ₂)·(∇_{r₂}J/J)]
        """
        cfg   = self.cfg
        d_phi = cfg.fdm_delta   # krok do numerycznego gradientu φ

        dr    = r1 - r2                                          # (N, 3)
        r12   = np.sqrt(np.sum(dr**2, axis=-1)) + 1e-30         # (N,)
        r_hat = dr / r12[:, None]                               # (N, 3), wersor r₁-r₂

        _, du_dr, d2u_dr2 = jastrow_and_derivs(r12, cfg.jastrow_alpha)

        # ─ Człon [B]: ∇²_{r₁}J/J  (= ∇²_{r₂}J/J, bo J zależy od |r₁-r₂|)
        #
        # Radialny laplasjian 3D: ∇² = d²/dr² + (2/r)·d/dr  →
        #   ∇²J/J = d²u/dr² + (du/dr)² + (2/r₁₂)·du/dr
        #
        lap_J_over_J_one = d2u_dr2 + du_dr**2 + (2.0 / r12) * du_dr   # (N,)

        # Dla r₂: ∇_{r₂}J = -∇_{r₁}J, więc (∇_{r₂})² ma ten sam laplasjian
        lap_J_over_J = 2.0 * lap_J_over_J_one   # oba elektrony

        # ─ Człon [C]: gradient φ/φ · gradient J/J (numeryczny)
        #
        # ∇_{r₁}J/J = (du/dr) · r̂₁₂,  shape (N, 3)
        grad_J_over_J_r1 = du_dr[:, None] * r_hat    # (N, 3)
        grad_J_over_J_r2 = -grad_J_over_J_r1         # ∇_{r₂}J/J = -∇_{r₁}J/J

        # Gradient φ(r)/φ(r) numerycznie — 6-kierunkowy stencil
        def grad_phi_over_phi(r):
            """∇φ/φ, shape (N,3) — metoda różnic skończonych."""
            phi_c = self.psi(r) + 1e-300
            grad  = np.zeros_like(r)
            for i, ax in enumerate([[d_phi,0,0],[0,d_phi,0],[0,0,d_phi]]):
                ax = np.array(ax)
                phi_plus  = self.psi(r + ax)
                phi_minus = self.psi(r - ax)
                grad[:, i] = (phi_plus - phi_minus) / (2 * d_phi * phi_c)
            return grad

        gphi1 = grad_phi_over_phi(r1)   # (N, 3)
        gphi2 = grad_phi_over_phi(r2)   # (N, 3)

        # Iloczyn skalarny gradientów
        cross = 2.0 * (
            np.sum(gphi1 * grad_J_over_J_r1, axis=-1) +
            np.sum(gphi2 * grad_J_over_J_r2, axis=-1)
        )  # (N,)

        return lap_J_over_J, cross

    # ── Główna metoda: E_L(r₁, r₂) ──────────────────────────────────────

    def __call__(self, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """
        Oblicza energię lokalną dla N walkerów jednocześnie.

        E_L = -ℏ²/(2m) · [ (∇²φ₁/φ₁) + (∇²φ₂/φ₂)        ← człon [A]×2
                           + (∇²_{r₁}J/J) + (∇²_{r₂}J/J)  ← człon [B]
                           + cross_term ]                    ← człon [C]
            + V_ext(r₁) + V_ext(r₂)
            + e²/(4πε·r₁₂)                                  ← Coulomb

        Argumenty:
            r1, r2 : shape (N_walkers, 3), pozycje elektronów [m]
        Zwraca:
            E_L : shape (N_walkers,), energie lokalne [J]
        """
        cfg  = self.cfg
        hb2m = HBAR**2 / (2.0 * cfg.m_eff)   # ℏ²/(2m*) [J·m²]

        # ─ Człon [A]: jednocząstkowe laplasjany ──────────────────────────
        lap1 = self._laplacian_phi_over_phi(r1)   # ∇²φ₁/φ₁, (N,)
        lap2 = self._laplacian_phi_over_phi(r2)   # ∇²φ₂/φ₂, (N,)

        # ─ Człony [B] i [C]: poprawki od Jastrowa ────────────────────────
        lap_J, cross = self._jastrow_kinetic_terms(r1, r2)

        # Pełny człon kinetyczny
        T = -hb2m * (lap1 + lap2 + lap_J + cross)   # (N,)

        # ─ Potencjał zewnętrzny (studnia kwantowa) ────────────────────────
        V1 = self.V_itp(r1)   # (N,)
        V2 = self.V_itp(r2)   # (N,)

        # ─ Odpychanie Coulomba ────────────────────────────────────────────
        # V_C = e² / (4πε₀εᵣ · r₁₂)
        # To jest FIZYCZNIE KLUCZOWE — bez tego mamy tylko 2×E_1cząstk.
        # VMC uwzględnia ten człon DOKŁADNIE (nie perturbacyjnie).
        dr   = r1 - r2
        r12  = np.sqrt(np.sum(dr**2, axis=-1)) + 1e-30
        eps  = cfg.eps_r * EPS_0
        V_C  = E_CHARGE**2 / (4.0 * np.pi * eps * r12)   # (N,)

        return T + V1 + V2 + V_C


# ─────────────────────────────────────────────────────────────────────────────
# ALGORYTM METROPOLISA-HASTINGSA (wektoryzowany)
# ─────────────────────────────────────────────────────────────────────────────

class MetropolisSampler:
    """
    Próbkuje z |Ψ(r₁,r₂)|² i zbiera próbki E_L.

    Dlaczego Metropolis a nie odrzucanie (rejection sampling)?
    → |Ψ|² jest znormalizowane, ale nie znamy stałej normalizacji.
      Metropolis wymaga tylko stosunku |Ψ_new|²/|Ψ_old|² → idealne.

    Dlaczego N_walkers konfiguracji jednocześnie?
    → Python ma wolne pętle. NumPy operuje na tablicach błyskawicznie.
      N_walkers = 2048 → 2048-krotne przyspieszenie względem pętli po walkerach.
    """

    def __init__(self, trial_wf: TrialWavefunction,
                 local_energy: LocalEnergy, cfg: VMCConfig):
        self.wf  = trial_wf
        self.EL  = local_energy
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=42)   # reprodukowalność

    def _init_walkers(self, x_range, y_range, z_range) -> np.ndarray:
        """
        Losowe startowe pozycje walkerów — równomiernie wewnątrz siatki.
        Konfiguracje z |Ψ| ≈ 0 (poza cropką) przesuwamy do centrum.

        Zwraca: shape (N_walkers, 2, 3) — N konfiguracji, 2 elektrony, 3 wsp.
        """
        N  = self.cfg.n_walkers
        lo = np.array([x_range[0], y_range[0], z_range[0]])
        hi = np.array([x_range[1], y_range[1], z_range[1]])

        r1 = self.rng.uniform(lo, hi, size=(N, 3))
        r2 = self.rng.uniform(lo, hi, size=(N, 3))

        # Sprawdź czy walkery mają sensowne |Ψ|
        psi_vals = np.abs(self.wf(r1, r2))
        mask = psi_vals < 1e-20
        if mask.any():
            center = 0.5 * (lo + hi)
            noise  = self.rng.uniform(-1e-10, 1e-10, (mask.sum(), 3))
            r1[mask] = center + noise
            r2[mask] = center + self.rng.uniform(-1e-10, 1e-10, (mask.sum(), 3))

        return np.stack([r1, r2], axis=1)   # (N, 2, 3)

    def _step(self, walkers: np.ndarray,
              log_prob_old: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Jeden krok Metropolisa dla WSZYSTKICH walkerów jednocześnie.

        Propozycja: r' = r + δ·η,  η ~ N(0,1) (symetryczna → Metropolis, nie M-H)
        Acceptance ratio: A = |Ψ(r'₁,r'₂)|²/|Ψ(r₁,r₂)|² = exp(log|Ψ'|² - log|Ψ|²)

        W przestrzeni log: akceptuj jeśli log(A) > log(U), U ~ Uniform[0,1].
        """
        N = self.cfg.n_walkers
        s = self.cfg.step_size

        proposal     = walkers + s * self.rng.standard_normal((N, 2, 3))
        r1_new, r2_new = proposal[:, 0, :], proposal[:, 1, :]

        log_prob_new = self.wf.log_abs_sq(r1_new, r2_new)
        log_ratio    = log_prob_new - log_prob_old           # log A
        accept       = log_ratio >= np.log(self.rng.uniform(0, 1, N) + 1e-300)

        # Wektorowa aktualizacja: gdzie accept=False, zostaw stare
        walkers_new  = np.where(accept[:, None, None], proposal, walkers)
        log_prob_out = np.where(accept, log_prob_new, log_prob_old)

        return walkers_new, log_prob_out, float(accept.mean())

    def run(self, x_range, y_range, z_range) -> dict:
        """
        Główna pętla VMC.

        Faza 1 — Termalizacja (warmup):
            Walkerzy "zapominają" o startowych pozycjach i zbiegają
            do obszarów o dużym |Ψ|². Wyniki tej fazy są odrzucane.
            Adaptacyjna kalibracja δ dąży do target acceptance rate.

        Faza 2 — Produkcja:
            Zbieramy E_L(r₁,r₂) dla każdego kroku.
            <E> = mean(E_L),  σ_E = std(E_L)/√N_eff

        Zwraca dict z wynikami w eV.
        """
        cfg = self.cfg

        walkers  = self._init_walkers(x_range, y_range, z_range)
        r1, r2   = walkers[:, 0, :], walkers[:, 1, :]
        log_prob = self.wf.log_abs_sq(r1, r2)

        # ── Termalizacja ─────────────────────────────────────────────────
        if cfg.verbose:
            print(f"🔥 Termalizacja: {cfg.n_warmup} kroków × {cfg.n_walkers} walkerów")

        for step in range(cfg.n_warmup):
            walkers, log_prob, ar = self._step(walkers, log_prob)

            if cfg.adaptive_step and step % 50 == 49:
                # Zwiększaj krok gdy za często akceptujemy (za mały zasięg)
                # Zmniejszaj gdy zbyt rzadko (za duże skoki)
                if ar > cfg.target_ar + 0.05:
                    cfg.step_size *= 1.1
                elif ar < cfg.target_ar - 0.05:
                    cfg.step_size *= 0.9

        if cfg.verbose:
            print(f"   Krok po kalibracji: {cfg.step_size:.3e} m")

        # ── Produkcja ────────────────────────────────────────────────────
        if cfg.verbose:
            print(f"⚛️  Produkcja: {cfg.n_steps} kroków...")

        energies = []
        ar_acc   = []
        t0 = time.time()

        for _ in range(cfg.n_steps):
            walkers, log_prob, ar = self._step(walkers, log_prob)
            ar_acc.append(ar)

            r1, r2 = walkers[:, 0, :], walkers[:, 1, :]
            E_L    = self.EL(r1, r2)   # (N_walkers,)

            # Usuwamy skrajne wartości: E_L >> median wskazuje na walker
            # blisko węzła φ (dzielenie przez 0 w ∇²φ/φ)
            E_median = np.median(np.abs(E_L))
            E_L_clean = E_L[np.abs(E_L) < 10.0 * E_median]

            if len(E_L_clean) > 0:
                energies.append(E_L_clean.mean())

        elapsed  = time.time() - t0
        energies = np.array(energies)

        E_J    = energies.mean()
        E_eV   = E_J / E_CHARGE
        std_eV = energies.std() / E_CHARGE
        err_eV = std_eV / np.sqrt(len(energies))
        mean_ar = float(np.mean(ar_acc))

        if cfg.verbose:
            print(f"\n{'─'*52}")
            print(f"  <E>            = {E_eV:.4f} ± {err_eV:.4f} eV")
            print(f"  Odch. stand.   = {std_eV:.4f} eV")
            print(f"  Acceptance rate= {mean_ar:.1%}")
            print(f"  Czas produkcji = {elapsed:.1f} s")
            print(f"{'─'*52}")

        return {
            "energy_mean":     E_eV,
            "energy_std":      std_eV,
            "energy_err":      err_eV,
            "acceptance_rate": mean_ar,
            "energies_raw":    energies,   # [J], do dalszej analizy
            "elapsed_s":       elapsed,
        }


# ─────────────────────────────────────────────────────────────────────────────
# OPTYMALIZACJA PARAMETRU JASTROWA α
# ─────────────────────────────────────────────────────────────────────────────

def optimize_jastrow(base_cfg: VMCConfig, psi_itp: PsiInterpolator,
                     V_itp, x, y, z,
                     alpha_range=(0.5e9, 5e9), n_points=8) -> Tuple[float, float]:
    """
    Skanuje α w podanym zakresie i wybiera wartość minimalizującą <E>.

    Matematyczne uzasadnienie (Zasada Wariacyjna):
        Dla KAŻDEGO Ψ_trial:  <Ψ_trial|Ĥ|Ψ_trial> ≥ E_0
    Zatem minimalizacja <E> po α daje najlepsze przybliżenie energii stanu
    podstawowego w przyjętej rodzinie funkcji próbnych.

    Uwagi:
    - alpha_range w [1/m]: α ~ 1/(2·a_B*) gdzie a_B* = bohrowski promień
      efektywny w GaAs ≈ 10 nm → α ~ 5×10⁷ m⁻¹
    - Podczas skanowania używamy mniejszego n_steps dla szybkości

    Zwraca: (best_alpha, best_energy_eV)
    """
    from copy import deepcopy
    alphas   = np.linspace(alpha_range[0], alpha_range[1], n_points)
    energies = []

    print(f"\n🔍 Optymalizacja α Jastrowa ({n_points} punktów)...")

    x_range = (x[0], x[-1])
    y_range = (y[0], y[-1])
    z_range = (z[0], z[-1])

    for a in alphas:
        cfg             = deepcopy(base_cfg)
        cfg.jastrow_alpha = float(a)
        cfg.n_warmup    = 100
        cfg.n_steps     = 500
        cfg.verbose     = False

        trial   = TrialWavefunction(psi_itp, cfg)
        loc_e   = LocalEnergy(psi_itp, V_itp, cfg)
        sampler = MetropolisSampler(trial, loc_e, cfg)
        res     = sampler.run(x_range, y_range, z_range)

        energies.append(res["energy_mean"])
        print(f"   α = {a:.3e} m⁻¹  →  <E> = {res['energy_mean']:.4f} eV")

    best_i     = int(np.argmin(energies))
    best_alpha = float(alphas[best_i])
    print(f"✅ Optymalne α = {best_alpha:.3e} m⁻¹  (<E> = {energies[best_i]:.4f} eV)")
    return best_alpha, float(energies[best_i])


# ─────────────────────────────────────────────────────────────────────────────
# PUNKT WEJŚCIA — integracja z solverem FDM
# ─────────────────────────────────────────────────────────────────────────────

def _build_V_itp(V: np.ndarray, x, y, z):
    """Pomocnicza: buduje interpolator potencjału zewnętrznego."""
    return si.RegularGridInterpolator(
        (x, y, z), V,
        method="linear",
        bounds_error=False,
        fill_value=float(V.max()),   # poza siatką → bariera (najwyższy potencjał)
    )


def run_vmc(
    psi:    np.ndarray,   # jednocząstkowa f. falowa z FDM, shape (nx,ny,nz)
    V:      np.ndarray,   # potencjał zewnętrzny [J], shape (nx,ny,nz)
    x:      np.ndarray,   # oś x [m]
    y:      np.ndarray,   # oś y [m]
    z:      np.ndarray,   # oś z [m]
    eigval: float,        # energia własna 1-cząstkowa [J] (do porównania)
    cfg:    Optional[VMCConfig] = None,
    optimize_alpha: bool = False,   # True = wolniej, ale lepsza α
) -> dict:
    """
    Główna funkcja — wywołaj po rozwiązaniu równania Schrödingera FDM.

    Interpretacja wyniku:
        result['energy_mean']  = <E₂e>  [eV]  — energia układu dwóch elektronów
        2 × eigval_eV          = 2×E₁e  [eV]  — energia bez korelacji
        Różnica                           [eV]  — korekta korelacyjna (zawiera
                                                  ΔT_korel + ΔV_C)
    """
    if cfg is None:
        cfg = VMCConfig()

    x_range = (float(x[0]), float(x[-1]))
    y_range = (float(y[0]), float(y[-1]))
    z_range = (float(z[0]), float(z[-1]))

    psi_itp = PsiInterpolator(psi, x, y, z)
    V_itp   = _build_V_itp(V, x, y, z)

    if optimize_alpha:
        best_alpha, _ = optimize_jastrow(cfg, psi_itp, V_itp, x, y, z)
        cfg.jastrow_alpha = best_alpha

    trial   = TrialWavefunction(psi_itp, cfg)
    loc_e   = LocalEnergy(psi_itp, V_itp, cfg)
    sampler = MetropolisSampler(trial, loc_e, cfg)

    eigval_eV = eigval / E_CHARGE

    if cfg.verbose:
        print(f"\n{'═'*52}")
        print(f"  VMC — GaAs/AlAs Quantum Dot, 2 elektrony")
        print(f"  E₁e (FDM):   {eigval_eV:.4f} eV")
        print(f"  2×E₁e:       {2*eigval_eV:.4f} eV  (bez korelacji)")
        print(f"  Walkerów:    {cfg.n_walkers}")
        print(f"  Kroków:      {cfg.n_steps}")
        print(f"  α Jastrowa:  {cfg.jastrow_alpha:.3e} m⁻¹")
        print(f"{'═'*52}\n")

    result = sampler.run(x_range, y_range, z_range)
    result["eigval_1p_eV"]  = eigval_eV
    result["jastrow_alpha"] = cfg.jastrow_alpha

    if cfg.verbose:
        corr = result["energy_mean"] - 2.0 * eigval_eV
        print(f"\n🎯 Energia 2e:     {result['energy_mean']:.4f} ± {result['energy_err']:.4f} eV")
        print(f"   Korekta korel.: {corr:+.4f} eV  (= V_Coulomb + ΔT_kinet)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SZYBKI TEST (bez solvera FDM — gaussowska funkcja próbna)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test VMC (Gaussian mock, bez solvera FDM) ===\n")

    nx = ny = nz = 30
    dx = 0.5e-9
    x = (np.arange(nx) - nx//2) * dx
    y = (np.arange(ny) - ny//2) * dx
    z = (np.arange(nz) - nz//2) * dx
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    sigma    = 5e-9
    psi_mock = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    psi_mock /= np.sqrt(np.sum(psi_mock**2) * dx**3)

    V_mock  = np.zeros_like(psi_mock)
    outside = (X**2 + Y**2 + Z**2) > (10*dx)**2
    V_mock[outside] = 1.0 * E_CHARGE   # bariera 1 eV

    cfg = VMCConfig(
        n_walkers=256, n_steps=300, n_warmup=100,
        jastrow_alpha=1e8,   # typowe α dla GaAs w jednostkach SI [1/m]
        verbose=True,
    )