"""
ci_engine.py — Configuration Interaction dla dwóch elektronów w kropce kwantowej
==================================================================================

IDEA OGÓLNA (dla matematyka)
==============================

Mamy Hamiltonian dwuciałowy:

    Ĥ = ĥ(r₁) + ĥ(r₂) + V_C(r₁,r₂)

gdzie:
    ĥ(r)      = -ℏ²/(2m*)·∇² + V_ext(r)    jednocząstkowy operator (z FDM)
    V_C(r₁,r₂) = e²/(4πε·|r₁-r₂|)          odpychanie Coulomba

Szukamy energii stanu podstawowego E_2 układu dwóch elektronów.

METODA: Configuration Interaction (CI)
========================================

Zamiast rozwiązywać równanie dla Ψ(r₁,r₂) w przestrzeni 6D,
rozwijamy Ψ w skończonej bazie funkcji dwucząstkowych:

    Ψ(r₁,r₂) = Σ_{mn} c_{mn} · Φ_{mn}(r₁,r₂)

gdzie Φ_{mn} to antysymetryczne iloczyny funkcji jednocząstkowych
(determinanty Slatera — spełniają zasadę Pauliego).

Wtedy zadanie na wartości własne Ĥ|Ψ⟩ = E|Ψ⟩ redukuje się do
zwykłego problemu algebraicznego:

    H·c = E·c

gdzie macierz H_{(mn),(kl)} = ⟨Φ_{mn}|Ĥ|Φ_{kl}⟩.

BAZA DWUCZĄSTKOWA
==================

Dwa elektrony mają spin ½. Stan singletowy (spin antyrównoległy, S=0)
ma symetryczną część orbitalną:

    Φ_{mn}(r₁,r₂) = [φ_m(r₁)φ_n(r₂) + φ_n(r₁)φ_m(r₂)] / N_{mn}

gdzie N_{mn} = √2 dla m≠n, N_{mm} = 2.

Stan trypletowy (S=1) ma antysymetryczną część orbitalną:

    Φ_{mn}(r₁,r₂) = [φ_m(r₁)φ_n(r₂) - φ_n(r₁)φ_m(r₂)] / √2

Tutaj liczymy SINGLET (stan podstawowy dla niezależnych elektronów to m=n=0,
co jest możliwe tylko w singlecie — tryplety wymagają m≠n).

ELEMENTY MACIERZY H
====================

Dzięki ortogonalności jednocząstkowych φ_n:

  Człon jednocząstkowy (diagonal w bazie CI):

    ⟨Φ_{mn}|ĥ₁+ĥ₂|Φ_{kl}⟩ = (ε_m + ε_n)·δ_{mk}δ_{nl}

  gdzie ε_n = energia własna n-tego stanu z FDM.

  Człon Coulomba — TUTAJ WCHODZI MC:

    ⟨Φ_{mn}|V_C|Φ_{kl}⟩ = J_{mnkl} ± K_{mnkl}

  gdzie całki kulombowskie (bezpośrednia J i wymienna K):

    J_{mnkl} = ∫∫ φ_m(r₁)φ_n(r₂) · [e²/4πε·r₁₂] · φ_k(r₁)φ_l(r₂) dr₁dr₂

    K_{mnkl} = ∫∫ φ_m(r₁)φ_n(r₂) · [e²/4πε·r₁₂] · φ_l(r₁)φ_k(r₂) dr₁dr₂

  Dokładny wzór na element macierzowy H dla singletów:
    H_{(mn),(kl)} = (ε_m+ε_n)·δ + J_{mnkl} + K_{mnkl}   (+ dla singletów)
    H_{(mn),(kl)} = (ε_m+ε_n)·δ + J_{mnkl} - K_{mnkl}   (- dla trypletów)

CAŁKI COULOMBA METODĄ MONTE CARLO
====================================

J_{mnkl} = ∫∫ f(r₁,r₂) dr₁dr₂,  gdzie f = φ_m·φ_n·V_C·φ_k·φ_l

Zamiast całkować po równomiernej siatce 6D (niemożliwe dla 50³ siatki),
używamy importance sampling:

  Gęstość próbkowania:  p(r₁,r₂) ∝ |φ_m(r₁)|·|φ_k(r₁)| · |φ_n(r₂)|·|φ_l(r₂)|

  Estymator MC:
    J_{mnkl} ≈ (1/N) Σᵢ [ φ_m(r₁ⁱ)φ_n(r₂ⁱ)·V_C(r₁₂ⁱ)·φ_k(r₁ⁱ)φ_l(r₂ⁱ) / p(r₁ⁱ,r₂ⁱ) ]

  Próbkujemy r₁ z |φ_m·φ_k| i r₂ z |φ_n·φ_l| niezależnie.
  Normalizacja gęstości nie jest potrzebna — wchodzi do mianownika.

WYNIK KOŃCOWY
==============

    E_1 = ε_0           (energia jednocząstkowa stanu podstawowego z FDM)
    E_2 = min eigenval(H_CI)    (energia dwucząstkowa z CI)
    E_X = E_2 - 2·E_1   (energia korelacji = efekt Coulomba)
"""

import numpy as np
import scipy.interpolate as si
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import itertools


# ─────────────────────────────────────────────────────────────────────────────
# STAŁE FIZYCZNE
# ─────────────────────────────────────────────────────────────────────────────

E_CHARGE = 1.602176634e-19   # [C]
HBAR     = 1.054571817e-34   # [J·s]
EPS_0    = 8.8541878128e-12  # [F/m]


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CIConfig:
    """
    Parametry obliczenia CI.

    n_orbitals : ile jednocząstkowych stanów φ_n wchodzi do bazy CI.
                 Liczba całek Coulomba rośnie jak n_orbitals⁴, więc:
                   6  stanów →   81 unikalnych całek  (szybko)
                   10 stanów →  625 całek             (kilkanaście sekund)

    n_mc_samples : ile punktów MC na jedną całkę Coulomba.
                   Błąd całki spada jak 1/√N.
                   Zalecane: 50_000 – 500_000.

    eps_r : przenikalność względna materiału (GaAs: 12.9)

    m_eff : masa efektywna [kg] (GaAs: 0.067·mₑ)

    spin_state : 'singlet' (S=0) lub 'triplet' (S=1).
                 Stan podstawowy 2e to zwykle singlet.

    seed : ziarno generatora losowego (reprodukowalność)
    """
    n_orbitals:   int   = 10
    n_mc_samples: int   = 100_000
    eps_r:        float = 12.9
    m_eff:        float = 0.067 * 9.1093837015e-31
    spin_state:   str   = 'singlet'   # 'singlet' lub 'triplet'
    seed:         int   = 42
    verbose:      bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# INTERPOLATORY FUNKCJI JEDNOCZĄSTKOWYCH
# ─────────────────────────────────────────────────────────────────────────────

class OrbitalSet:
    """
    Przechowuje i interpoluje zestaw jednocząstkowych funkcji falowych φ_n(r).

    Wejście:
        eigvecs : array (N_grid, n_orbitals) — kolumny to kolejne φ_n
                  (takiej postaci zwraca scipy.sparse.linalg.eigsh)
        eigvals : array (n_orbitals,) — odpowiadające energie [J]
        x, y, z : 1D arrays — siatka [m]
        nx,ny,nz: wymiary siatki

    Użycie:
        orbs = OrbitalSet(eigvecs, eigvals, x, y, z, nx, ny, nz)
        phi_0 = orbs(r, 0)   # wartości φ_0 w punktach r, shape (N,)
        phi_3 = orbs(r, 3)   # wartości φ_3 w punktach r, shape (N,)
    """

    def __init__(self, eigvecs: np.ndarray, eigvals: np.ndarray,
                 x, y, z, nx, ny, nz):
        self.eigvals = eigvals          # energie własne [J]
        self.n = eigvecs.shape[1]       # liczba orbitali
        self._itps = []                 # lista interpolatorów

        for k in range(self.n):
            psi_k = eigvecs[:, k].reshape(nx, ny, nz).real
            itp = si.RegularGridInterpolator(
                (x, y, z), psi_k,
                method='linear',
                bounds_error=False,
                fill_value=0.0,
            )
            self._itps.append(itp)

    def __call__(self, r: np.ndarray, n: int) -> np.ndarray:
        """
        r : shape (..., 3)
        n : indeks orbitalu
        zwraca: φ_n(r), shape (...)
        """
        shape_in = r.shape[:-1]
        vals = self._itps[n](r.reshape(-1, 3))
        return vals.reshape(shape_in)

    def energy(self, n: int) -> float:
        """Energia własna ε_n [J]."""
        return float(self.eigvals[n])


# ─────────────────────────────────────────────────────────────────────────────
# CAŁKI COULOMBA METODĄ MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

class CoulombIntegralsMC:
    """
    Liczy całki kulombowskie metodą crude Monte Carlo:

        J_{mnkl} = ∫∫ φ_m(r₁)·φ_n(r₂) · [e²/(4πε·r₁₂)] · φ_k(r₁)·φ_l(r₂) dr₁dr₂

    Strategia: crude MC z rozkładu jednostajnego na pudełku
    ─────────────────────────────────────────────────────────
    Losujemy r₁, r₂ równomiernie z pudełka o objętości V_box = V_x · V_y · V_z.
    Estymator:

        J_{mnkl} ≈ V_box² · (1/N) Σᵢ f(r₁ⁱ, r₂ⁱ)

    gdzie f(r₁,r₂) = φ_m(r₁)·φ_n(r₂)·V_C(r₁₂)·φ_k(r₁)·φ_l(r₂).

    Błąd statystyczny: σ/√N, gdzie σ = std(f)·V_box².

    Dlaczego nie importance sampling?
    ──────────────────────────────────
    IS wymaga próbkowania z gęstości p ∝ |f|, co dla iloczynu czterech
    funkcji na siatce 3D jest numerycznie skomplikowane i łatwo o błędy
    w normalizacji (dokładnie to powodowało ujemne wartości wyżej).
    Crude MC jest prostszy, deterministycznie poprawny i wystarczająco
    szybki dla funkcji skoncentrowanych w małej kropce.

    Symetrie całek (redukują liczbę obliczeń):
    ───────────────────────────────────────────
    W "chemist notation" J(a,b,c,d) = ∫∫ φ_a(1)φ_b(1)·V_C·φ_c(2)φ_d(2):

        J(a,b,c,d) = J(b,a,c,d) = J(a,b,d,c) = J(c,d,a,b)

    Implementujemy klucz kanoniczny: para (min(a,b),max(a,b)) × (min(c,d),max(c,d)),
    a następnie sortujemy obie pary leksykograficznie.
    """

    def __init__(self, orbs: OrbitalSet, cfg: CIConfig,
                 x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.orbs = orbs
        self.cfg  = cfg
        self.rng  = np.random.default_rng(cfg.seed)
        self.eps  = cfg.eps_r * EPS_0

        # Pudełko całkowania = siatka FDM
        self.lo  = np.array([x[0],  y[0],  z[0]])
        self.hi  = np.array([x[-1], y[-1], z[-1]])
        self.vol = float(np.prod(self.hi - self.lo))   # V_box [m³]

        self._cache: dict = {}   # canonical_key → wartość całki [J]

    # ── Klucz kanoniczny ─────────────────────────────────────────────────

    @staticmethod
    def _key(a: int, b: int, c: int, d: int) -> tuple:
        """
        Kanoniczny klucz dla J(a,b,c,d) w "chemist notation".

        Symetrie:
            J(a,b,c,d) = J(b,a,c,d)   ← zamiana elektronów w orbitalu 1
            J(a,b,c,d) = J(a,b,d,c)   ← zamiana elektronów w orbitalu 2
            J(a,b,c,d) = J(c,d,a,b)   ← zamiana obu par (r₁↔r₂)
        """
        p1 = (min(a, b), max(a, b))
        p2 = (min(c, d), max(c, d))
        return (min(p1, p2), max(p1, p2))

    # ── Jedna całka Coulomba (crude MC) ──────────────────────────────────

    def compute(self, a: int, b: int, c: int, d: int) -> float:
        """
        Oblicza J(a,b,c,d) = ∫∫ φ_a(r₁)φ_b(r₁)·V_C(r₁₂)·φ_c(r₂)φ_d(r₂) dr₁dr₂

        Estymator crude MC:
            J ≈ V_box² · mean_i[ φ_a(r₁ⁱ)φ_b(r₁ⁱ)·V_C(r₁₂ⁱ)·φ_c(r₂ⁱ)φ_d(r₂ⁱ) ]
            r₁ⁱ, r₂ⁱ ~ Uniform(pudełko)

        Zwraca wartość całki w [J].
        """
        key = self._key(a, b, c, d)
        if key in self._cache:
            return self._cache[key]

        N   = self.cfg.n_mc_samples

        # Losuj punkty równomiernie z pudełka
        r1 = self.rng.uniform(self.lo, self.hi, size=(N, 3))   # (N, 3)
        r2 = self.rng.uniform(self.lo, self.hi, size=(N, 3))   # (N, 3)

        # Wartości orbitali
        phi_a = self.orbs(r1, a)   # (N,)
        phi_b = self.orbs(r1, b)   # (N,)
        phi_c = self.orbs(r2, c)   # (N,)
        phi_d = self.orbs(r2, d)   # (N,)

        # Odległość r₁₂ i potencjał Coulomba
        r12 = np.sqrt(np.sum((r1 - r2)**2, axis=-1)) + 1e-30
        V_C = E_CHARGE**2 / (4.0 * np.pi * self.eps * r12)    # (N,) [J]

        # Podcałkowa
        f = phi_a * phi_b * V_C * phi_c * phi_d               # (N,)

        # Estymator: V_box² · mean(f)
        result = float(np.mean(f)) * self.vol**2

        self._cache[key] = result
        return result

    def get(self, a: int, b: int, c: int, d: int) -> float:
        """Odczyt z cache — wywoływać po compute_all()."""
        return self._cache.get(self._key(a, b, c, d), 0.0)

    # ── Oblicz wszystkie potrzebne całki ─────────────────────────────────

    def compute_all(self, n_orb: int) -> None:
        """
        Oblicza wszystkie unikalne J(a,b,c,d) dla a,b,c,d ∈ [0, n_orb).

        Iteruje tylko po kanoniczych kombinacjach, pomijając duplikaty.
        Liczba unikalnych całek = n_orb²(n_orb²+1)/8 + ...
        W praktyce dla n_orb=10: ~210 unikalnych całek.
        """
        if self.cfg.verbose:
            print(f"\n🔢 Obliczam całki Coulomba (crude MC)...")
            print(f"   Orbitale: {n_orb}")
            print(f"   Próbek MC na całkę: {self.cfg.n_mc_samples:,}")

        t0    = time.time()
        count = 0

        # Iteruj po unikalnych kluczach kanonicznych
        # para (a,b): a ≤ b;  para (c,d): c ≤ d;  para1 ≤ para2 leksykograficznie
        pairs = [(a, b) for a in range(n_orb) for b in range(a, n_orb)]

        for i, (a, b) in enumerate(pairs):
            for (c, d) in pairs[i:]:      # para2 ≥ para1 → bez duplikatów
                key = self._key(a, b, c, d)
                if key not in self._cache:
                    self.compute(a, b, c, d)
                    count += 1

        elapsed = time.time() - t0
        if self.cfg.verbose:
            print(f"   Obliczono {count} unikalnych całek w {elapsed:.1f} s")


# ─────────────────────────────────────────────────────────────────────────────
# BAZA CI I MACIERZ HAMILTONIANU
# ─────────────────────────────────────────────────────────────────────────────

class CIBasis:
    """
    Buduje bazę dwucząstkową i macierz Hamiltonianu CI.

    Baza singletowa (S=0, stany symetryczne orbitalnie):
        |mn⟩ = [φ_m(r₁)φ_n(r₂) + φ_n(r₁)φ_m(r₂)] / N_{mn}
        N_{mn} = 2 dla m=n, √2 dla m≠n

    Elementy macierzy H:

      Człon jednocząstkowy (korzystamy z ĥ|φ_n⟩ = ε_n|φ_n⟩):

        ⟨mn|ĥ₁+ĥ₂|kl⟩ = (ε_m + ε_n) · δ_{mk}δ_{nl}

      Człon Coulomba dla singletów (m≤n, k≤l):

        ⟨mn|V_C|kl⟩ = J_{mknl} + J_{mlnk}

        gdzie J_{abcd} = ∫∫ φ_a(r₁)φ_b(r₁)·[e²/4πε·r₁₂]·φ_c(r₂)φ_d(r₂) dr₁dr₂

      Uwaga o indeksach: w literaturze spotykasz różne konwencje.
      Tutaj używamy "chemist notation": ⟨12|12⟩ = J_{1122}
    """

    def __init__(self, n_orbitals: int, spin_state: str = 'singlet'):
        self.n_orb = n_orbitals
        self.spin  = spin_state

        # Generuj pary (m,n) tworzące bazę
        self.pairs = self._build_pairs()
        self.dim   = len(self.pairs)

    def _build_pairs(self) -> List[Tuple[int, int]]:
        """
        Zwraca listę par (m,n) tworzących bazę.

        Singlet: m ≤ n (para m=n = oba elektrony w tym samym orbitalu)
        Tryplet: m < n (zakaz Pauliego → m≠n)
        """
        pairs = []
        for m in range(self.n_orb):
            start = m if self.spin == 'singlet' else m + 1
            for n in range(start, self.n_orb):
                pairs.append((m, n))
        return pairs

    def normalization(self, m: int, n: int) -> float:
        """
        Czynnik normalizacji stanu dwucząstkowego.

        ⟨mn|mn⟩ = 1 wymaga N_{mn} = √2 (m≠n) lub 2 (m=n, tylko singlet).
        """
        return 2.0 if m == n else np.sqrt(2.0)

    def build_hamiltonian(self, orbs: OrbitalSet,
                          coulomb: CoulombIntegralsMC) -> np.ndarray:
        """
        Buduje pełną macierz H_{CI} w bazie dwucząstkowej.

        Zwraca: array (dim, dim), rzeczywista symetryczna macierz [J]
        """
        dim = self.dim
        H   = np.zeros((dim, dim))

        for a, (m, n) in enumerate(self.pairs):
            N_mn = self.normalization(m, n)

            for b, (k, l) in enumerate(self.pairs):
                N_kl = self.normalization(k, l)

                # ── Człon jednocząstkowy ──────────────────────────────
                # ⟨Φ_{mn}|ĥ₁+ĥ₂|Φ_{kl}⟩ = (ε_m+ε_n)·δ_{ab}
                if a == b:
                    H[a, b] += orbs.energy(m) + orbs.energy(n)

                # ── Człon Coulomba ────────────────────────────────────
                #
                # "Chemist notation": J(a,b,c,d) = ∫∫ φ_a(1)φ_b(1)·V_C·φ_c(2)φ_d(2)
                #
                # Rozpisując ⟨Φ_{mn}|V_C|Φ_{kl}⟩:
                # = (1/(N_mn·N_kl)) × [J(m,k,n,l) + J(m,l,n,k)
                #                     + J(n,k,m,l) + J(n,l,m,k)]
                # Symetria J(a,b,c,d)=J(b,a,c,d) daje:
                #   = (2/(N_mn·N_kl)) × [J(m,k,n,l) + J(m,l,n,k)]

                J_direct   = coulomb.get(m, k, n, l)
                J_exchange = coulomb.get(m, l, n, k)
                prefactor  = 2.0 / (N_mn * N_kl)

                if self.spin == 'singlet':
                    H[a, b] += prefactor * (J_direct + J_exchange)
                else:
                    H[a, b] += prefactor * (J_direct - J_exchange)

        return H


# ─────────────────────────────────────────────────────────────────────────────
# DIAGONALIZACJA I WYNIK KOŃCOWY
# ─────────────────────────────────────────────────────────────────────────────

def solve_ci(H: np.ndarray, cfg: CIConfig, orbs: OrbitalSet) -> dict:
    """
    Diagonalizuje macierz H_CI i oblicza energię korelacji.

    Zwraca słownik z wynikami w eV.
    """
    # Pełna diagonalizacja (macierz mała: dim ~ 55 dla 10 orbitali)
    eigenvalues = np.linalg.eigvalsh(H)   # eigvalsh dla macierzy hermitowskich

    E_2_J  = eigenvalues[0]                     # energia stanu podstawowego [J]
    E_1_J  = orbs.energy(0)                     # jednocząstkowe ε_0 [J]

    E_2_eV = E_2_J / E_CHARGE
    E_1_eV = E_1_J / E_CHARGE
    E_X_eV = E_2_eV - 2.0 * E_1_eV             # energia korelacji [eV]

    if cfg.verbose:
        print(f"\n{'═'*52}")
        print(f"  WYNIKI CI")
        print(f"{'─'*52}")
        print(f"  E_1 (jednocząstkowe ε₀)  = {E_1_eV:.4f} eV")
        print(f"  2·E_1 (bez oddziaływań)  = {2*E_1_eV:.4f} eV")
        print(f"  E_2 (stan podst. CI)     = {E_2_eV:.4f} eV")
        print(f"  E_X = E_2 - 2·E_1        = {E_X_eV:+.4f} eV  ← energia korelacji")
        print(f"{'═'*52}")
        print(f"\n  Kilka najniższych stanów dwucząstkowych:")
        for i, e in enumerate(eigenvalues[:5]):
            print(f"    [{i}]  {e/E_CHARGE:.4f} eV")

    return {
        "E_1_eV":        E_1_eV,
        "E_2_eV":        E_2_eV,
        "E_X_eV":        E_X_eV,
        "eigenvalues_eV": eigenvalues / E_CHARGE,
        "H_CI":          H,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUNKT WEJŚCIA — kompatybilny z Twoim mainem
# ─────────────────────────────────────────────────────────────────────────────

def run_ci(
    eigvals:  np.ndarray,   # energie własne z FDM [J], shape (k,)
    eigvecs:  np.ndarray,   # wektory własne z FDM,  shape (N_grid, k)
    x: np.ndarray,          # siatka x [m]
    y: np.ndarray,          # siatka y [m]
    z: np.ndarray,          # siatka z [m]
    nx: int, ny: int, nz: int,
    cfg: Optional[CIConfig] = None,
) -> dict:
    """
    Główna funkcja — wywołaj z main() po rozwiązaniu FDM.

    Przykład użycia w main():

        from ci_engine import run_ci, CIConfig

        cfg = CIConfig(
            n_orbitals   = 10,
            n_mc_samples = 100_000,
            verbose      = True,
        )
        result = run_ci(eigvals, eigvecs, x, y, z, nx, ny, nz, cfg)
        print(f"E_X = {result['E_X_eV']:.4f} eV")
    """
    if cfg is None:
        cfg = CIConfig()

    n_orb = min(cfg.n_orbitals, eigvecs.shape[1])

    if cfg.verbose:
        print(f"\n{'═'*52}")
        print(f"  CI — Configuration Interaction, 2 elektrony")
        print(f"  Orbitale w bazie:  {n_orb}")
        print(f"  Wymiar bazy CI:    ", end="")

    # ── Zbuduj orbitale ──────────────────────────────────────────────────
    orbs = OrbitalSet(eigvecs[:, :n_orb], eigvals[:n_orb], x, y, z, nx, ny, nz)

    basis = CIBasis(n_orb, spin_state=cfg.spin_state)
    if cfg.verbose:
        print(f"{basis.dim}")
        print(f"  Stan spinowy:      {cfg.spin_state}")
        print(f"{'═'*52}")

    # ── Oblicz całki Coulomba MC ─────────────────────────────────────────
    coulomb = CoulombIntegralsMC(orbs, cfg, x, y, z)
    coulomb.compute_all(n_orb)

    # ── Zbuduj i rozwiąż macierz CI ──────────────────────────────────────
    if cfg.verbose:
        print(f"\n🔧 Buduję macierz H_CI ({basis.dim}×{basis.dim})...")

    H = basis.build_hamiltonian(orbs, coulomb)

    if cfg.verbose:
        print(f"   Macierz zbudowana. Diagonalizuję...")

    result = solve_ci(H, cfg, orbs)
    return result