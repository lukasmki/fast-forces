# Potential Reference

A self-contained statement of the potential energy functions used by this force
field: every functional form, every parameter, every constant.  Nothing here
refers to code — this is the record of *what* is evaluated, not how.

The model is a multi-state valence-bond Hamiltonian.  Each state `s` is one
bonding pattern (one bond graph) over the same set of nuclei.  The diagonal
`H_ss` is the potential energy of that bonding pattern; the off-diagonal `H_st`
couples two of them.  The physical energy is the lowest eigenvalue of `H`, and
forces follow by Hellmann–Feynman.

```
H_ss  =  E_bonded  +  E_ZBL  +  E_12-6  +  E_Coulomb  -  E_excl

H_st  =  V(x)                one Gaussian per reaction channel, nonzero only
                             near the geometry it is centred on
```

Only `E_bonded` and `E_excl` depend on the bonding pattern.  The three
whole-system pair sums `E_ZBL`, `E_12-6` and `E_Coulomb` are functions of the
nuclear positions and atomic numbers alone, with no reference to the bond graph,
and therefore take the same value on every state.

## Units and conventions

| quantity | unit |
| --- | --- |
| length | Å (`switch_radius`, `switch_width` and the bonded parameters below are stated in nm) |
| energy | eV (bonded parameters are stated in kJ/mol) |
| angle | radians |
| force | eV/Å |
| stress | eV/Å³ |

Dimensionless: `shape_decay`, `b`, `c`, `core_fraction`, `n`, `accuracy`.
`SCREENING_LENGTH` is Å; `ccoul` is eV·Å; `gamma` is 1/Å.  A coupling amplitude
`A` is eV and its width `a` is 1/Å²; `r0`, `ra0`, `rb0` are Å and `t0` radians.

Gradient conventions, with `v` any interatomic displacement vector:

```
F_i     = -dE/d(pos_i)
W_ab    =  sum_v  v_a * (dE/dv)_b              virial (3x3, symmetric)
stress  =  W / V
```

Every diagonal energy is a function of minimum-image displacement vectors alone,
so a homogeneous strain acts as `v -> (I + e) v` and the virial follows from the
same pair gradients as the forces.  A coupling is written over absolute
positions; its virial is `W_ab = -sum_i pos_a F_b`, which is origin-independent
because every coupling's forces sum to zero.

---

## 1. Bonded terms

Per-molecule terms defined on the bond graph of a state.  Throughout,
`r` is a bond length, `θ` a bond angle, `φ` a dihedral, and

```
dr  = r - r0
dc  = cos θ - cos θ0
```

Angles enter only through their cosines, never as angles.

### 1.1 Bond — Morse with a one-sided shape term

```
Dw = D + bond_asymptote     if dr > 0
Dw = D                      if dr <= 0

a  = sqrt( k / (2 Dw) )
s  = a * max(dr, 0)

E  = Dw [ 1 - exp(-a dr) ]^2  -  D  +  Dw * c * s^3 * exp(-b s)
```

Parameters: `D` (well depth), `r0` (equilibrium length), `k` (force constant),
`c` (shape amplitude), `b` (shape decay).

- The `-D` offset places the minimum at `-D`, so the dissociated limit sits at
  `bond_asymptote` above zero while the minimum is unchanged.
- The branch at `dr = 0` is C2: the curvature there is `2 Dw a^2 = k` for either
  value of `Dw`, so no fitted frequency sees the join.
- The shape term is `O(s^3)`, leaving `D`, `r0` and the curvature at `dr = 0`
  untouched by `c`; it is identically zero on the compressed branch.
- `c` is bounded by a monotonicity limit `c <= c_max(b)`, tabulated in §7.2.
- Where `b` is absent from a parameter set it takes the global `shape_decay`.

### 1.2 Bond — harmonic (non-reactive alternative)

```
E = 0.5 * k * dr^2                 D unused
```

### 1.3 Angle — cosine harmonic

```
E = 0.5 * k * (cos θ - cos θ0)^2
```

### 1.4 Cross terms

| term | energy | parameters | atoms |
| --- | --- | --- | --- |
| bond–bond | `max( k dr1 dr2 , -10 )` | `r1_0, r2_0, k` | 4 (two bonds) |
| bond–angle | `max( k dr dc , -20 )` | `theta0, r0, k` | 5 (angle + bond) |
| angle–angle | `k dc1 dc2` | `theta1_0, theta2_0, k` | 6 (two angles) |

The two lower clips are floors on the energy in kJ/mol, applied to the raw
product; the gradient vanishes where a clip is active.

### 1.5 Dihedrals

The dihedral about a central bond `vb` is `φ = atan2(S, C)` with

```
axis = vb / |vb|
u, v = the outer bond vectors projected perpendicular to axis
S    = (axis × u) · v
C    = u · v
```

| term | energy | parameters | atoms |
| --- | --- | --- | --- |
| periodic dihedral | `k (1 + cos(n φ - φ0))` | `phi0, n, k` | 4 |
| dihedral–bond | `k dr (1 + cos(n φ - φ0))` | `+ r0` | 6 |
| dihedral–angle | `k dc (1 + cos(n φ - φ0))` | `+ theta0` | 7 |
| dihedral–angle–angle | `k dc1 dc2 (1 + cos(n φ - φ0))` | `+ theta0_1, theta0_2` | 4 |

### 1.6 Reference offset

```
E = E0                             constant; zero force, zero virial
```

A per-molecule additive shift placing different molecular templates on a common
absolute energy scale.  `E0` is the residual between the molecule's reference
atomization energy and the depth its Morse bonds already supply.

---

## 2. Short-range repulsion — tapered ZBL

Summed over every pair in the system; parameterized by atomic number alone, with
no free parameters.

```
a      = SCREENING_LENGTH / ( Z1^0.23 + Z2^0.23 )
x      = r / a
phi(x) = sum_k C_k exp(-B_k x)

C = (0.18175, 0.50986, 0.28022, 0.02817)
B = (3.19980, 0.94229, 0.40290, 0.20162)

f(r)   = 1 / ( 1 + exp( (r - taper_radius) / taper_width ) )
df/dr  = -f (1 - f) / taper_width

u(r)   = f(r) * zbl_ccoul * Z1 * Z2 * phi(x) / r
```

`f` is a Fermi switch that turns the term **off** above `taper_radius`.

```
SCREENING_LENGTH = 0.46850 Å        taper_radius = 1.5  Å
zbl_ccoul        = 14.399645 eV·Å   taper_width  = 0.12 Å
```

Retained fraction `f` at representative bond lengths: 0.998 (H–H, 0.741 Å),
0.989 (O–H, 0.958 Å), 0.918 (O–O, 1.208 Å).  The taper exists because the bare
ZBL form is fitted to keV nuclear stopping and is far too repulsive in the
1.5–3 Å range; untapered it adds +0.72 eV to a water-dimer hydrogen bond and
+251 kbar to a water box.

---

## 3. Dispersion and contact — switched 12-6

Summed over every pair.  Pair parameters are geometric means of the per-atom
ones: `σ_ij = sqrt(σ_i σ_j)`, `ε_ij = sqrt(ε_i ε_j)`.

```
g(r)  = 1 - 1 / ( 1 + exp( (r - switch_radius) / switch_width ) )
dg/dr = g (1 - g) / switch_width

r_e   = max( r, core_fraction * σ )

u_126(r) = 4 ε [ (σ/r_e)^12 - (σ/r_e)^6 ]
           + du/dr|_{r_e} * (r - core_fraction σ)      for r < core_fraction σ

u(r)  = g(r) * u_126(r)
```

`g` is the reflection of the ZBL taper at the same width: it turns the term
**on** above `switch_radius`.  The linear continuation below `core_fraction σ`
is the C1 tangent at the core radius; it bounds what a strongly compressed pair
can contribute (≈5700 eV rather than ≈1e21 eV at 0.024 Å).

```
switch_radius = 0.22 nm (2.2 Å)     core_fraction = 0.4
switch_width  = taper_width / 10
```

`switch_radius` is deliberately **not** equal to `taper_radius`.  The two
switches are not complementary: there is a gap from ~1.6 to ~2.0 Å in which both
terms are small and electrostatics alone carries the hydrogen bond.  §7.2 gives
the numbers behind this choice.

---

## 4. Electrostatics — ACKS2

Charges are not fixed parameters; they are the solution of a variational
electronegativity-equalization problem at each geometry.

### 4.1 Charge kernel

```
K_ij = erf( gamma * r_ij ) / r_ij            gamma = 2.0 / Å
```

This is the interaction of two Gaussian-smeared charges.  It is finite at
contact, `K -> 2 gamma / sqrt(pi) ≈ 2.257`, so charges saturate rather than
diverge as atoms approach.

Under open boundaries the kernel is evaluated on the nearest image only, and
`K_ii = 0`.  Under full periodicity it is the lattice sum, Ewald-split at
`kappa`:

```
K_ij = sum_n' [ erf(gamma |r_ij + n|) - erf(kappa |r_ij + n|) ] / |r_ij + n|
     + (4 pi / V) sum_{k != 0} exp( -k^2 / (4 kappa^2) ) / k^2 * cos(k . r_ij)
     - delta_ij * 2 kappa / sqrt(pi)
     + background,        background = -pi / (kappa^2 V)
```

Two properties are specific to the periodic form:

- `K_ii != 0`: an atom interacts with its own periodic images, and this
  self-image term enters the equilibration alongside the hardness.
- The `k = 0` term is omitted and its neutralizing background restored
  explicitly.  Omission alone is legal only for charge-neutral contractions
  (a constant added to every entry of `K` contributes `(sum_i q_i)^2` to the
  energy); the explicit background makes each individual `K_ij` well defined,
  which the exclusion of §5 requires because it contracts `K` against a
  non-neutral weight.

`kappa` and the reciprocal-space cutoff are both fixed by a single target
`accuracy`.  For a periodic system a 3D lattice sum is the right sum only under
full periodicity; a slab or wire falls back to the nearest-image kernel.

### 4.2 The charge solve

Per-atom parameters: `mu` (electronegativity), `eta` (hardness), `soft_amp` and
`soft_decay` (the softness kernel).  The stationary point of the ACKS2 energy
functional in the charges `Q` and the Kohn–Sham potentials `u` is a linear
system `A x = b` with `x = [Q, u, λ_tot, λ_KS]`:

```
        | K + 2 diag(eta)     -I        -1   0 |        | -mu |
  A  =  |      -I           -L_X         0  -1 |   b =  |  0  |
        |      -1^T            0         0   0 |        |  0  |
        |       0            -1^T        0   0 |        |  0  |

  X_ij = soft_amp_i * soft_amp_j * exp( -r_ij / tau_ij ),
         tau_ij = ( soft_decay_i + soft_decay_j ) / 2

  L_X  = diag( sum_j X_ij ) - X            (the graph Laplacian of X)
```

The last two rows impose `sum_i Q_i = 0` and `sum_i u_i = 0`.  Only the `K` and
`X` blocks depend on geometry.  The hardness is **added** to the Coulomb
diagonal rather than replacing it, since under periodicity `K_ii` is a physical
self-image interaction.

The solve uses the **unmasked** kernel over every pair, so the charges are a
function of positions and elements alone — independent of the bond graph, hence
identical on every state.  `A` carries the bare kernel while the energy below
carries `ccoul`, so `mu` and `eta` are in the units that convention implies
rather than in eV directly.

### 4.3 Energy and gradient

```
E = (ccoul / 2) * sum_ij  S_ij * Q_i * Q_j * K_ij        ccoul = 14.4 eV·Å
```

with `S` the exclusion screen of §5.2 (`S = 1` everywhere if no exclusions).
Because the charges move with the atoms, the gradient has a response term:

```
dE/dr  =  (dE/dr)|_Q  -  lam^T (dA/dr) x,        lam = A^-1 (dE/dx)
```

`A` is symmetric, so the adjoint uses the same factorization and the response
costs one extra solve rather than one per coordinate.  The screen belongs to
`dE/dx` only: `A` is the unscreened matrix, so `dA/dr` is contracted against an
unmasked weight.

---

## 5. Exclusions

The three pair sums of §2–§4 run over every pair with no reference to the bond
graph.  Pairs separated by `exclusion_depth = 3` bonds or fewer are then removed,
so that a molecule's internal geometry is set by its bonded terms.  This
subtraction is the *second* bond-graph-dependent contribution to `H_ss`, and the
only part of the nonbonded energy that differs between states.

### 5.1 Additive exclusions (repulsion and dispersion)

```
E_excl,12-6 = -u_126(r, σ_ij, ε_ij)
E_excl,ZBL  = -u_ZBL(r, Z1, Z2)
```

evaluated with the identical functional forms of §2 and §3, switches included,
so the cancellation is exact.  A 12-6 exclusion is present only for pairs whose
atoms both carry nonzero `σ` and `ε`; where either vanishes the whole-system 12-6
is identically zero for that pair and there is nothing to cancel.  ZBL admits no
such exemption, since it has no free parameters.

### 5.2 Coulomb exclusion

The charges come from a solve whose matrix contains the kernel, so a Coulomb
exclusion cannot simply be subtracted from the energy, and masking the kernel
inside the solve would make the charges a function of the bond graph — which is
disallowed, since the charges must be identical on every state (the measured
cost of violating this is 0.88 eV of dependence on the arbitrary choice of
reference state).  The charges are therefore solved once from the unmasked
kernel and the exclusion applied afterwards, in two complementary pieces:

```
per state   H_ss += -ccoul * sum_{(i,j) in excl(s)} Q_i Q_j K_ij

once        S_ij = 1 - sum_s w_s M_ij^s
            E    = (ccoul / 2) sum_ij S_ij Q_i Q_j K_ij
```

with `M^s` the exclusion mask of state `s` and `w_s` its ground-state weight.
The per-state term is what lets the exclusion influence which bonding pattern is
lower; the screened contraction is the whole-system energy.

Fractional entries of `S` are the normal case and are not an interpolation:
each state's correction is linear in its own mask and the contraction is linear
in its weight, so the Hellmann–Feynman sum `sum_s w_s dE_s/dr` collapses into a
single weight matrix.  A pair bonded in every state comes out at exactly `S = 0`;
a pair bonded in some of them is removed in proportion.  `S` is held fixed under
the derivative, as Hellmann–Feynman prescribes for eigenvector components.

**Physical content.** The Coulomb exclusion removes the intramolecular half of
the polarization response, which is worth about 0.07 eV of a hydrogen bond: the
water-dimer well moves from 0.1677 eV at 2.85 Å to 0.1024 eV at 2.91 Å when it is
applied.  Two approaching molecules polarize each other (`q_H` rises from
+0.30399 to +0.31264 at 2.85 Å) and the intramolecular Coulomb energy falls along
with the intermolecular one; booking that intramolecular gain is exactly what the
exclusion prevents.  `eta` is the lever that pays the depth back, since the
intermolecular term scales as `q²`.  The charges themselves are unaffected — an
isolated water still gives `q_H = +0.30399` — so no molecular dipole moves.

---

## 6. Coupling — the off-diagonal

One Gaussian per reaction channel, in Å and eV.  Three forms exist, and a
channel's own connectivity change selects which applies: one bond broken or
formed separating two fragments is a **fission**; one bond broken and one formed
sharing a common atom is a **transfer**; anything else takes the RMSD fallback.

| form | channel | V | parameters |
| --- | --- | --- | --- |
| two-body | fission | `A exp(-a (r - r0)²)` | `A, a, r0` |
| three-body | transfer | `A exp(-a g)` | `A, a, ra0, rb0, t0` |
| RMSD | anything else | `(A/M) Σ_m exp(-a ρ_m²)` | `A, a` + a TS ensemble |

**Two-body.** `r` is the length of the single bond that changes.

**Three-body.** On the ordered triple (donor `D`, transferring atom `H`,
acceptor `A`) with the mover central:

```
ra = |H - D|,   rb = |H - A|,   d = |A - D|
d0 = sqrt( ra0² + rb0² - 2 ra0 rb0 cos t0 )        law of cosines
g  = (ra - ra0)² + (rb - rb0)² + (d - d0)²         Å²
```

`t0` is stored as an angle because that is the readable parameter, and converted
to a length so that one width `a` is dimensionally consistent across all three
sides.  The three sides are a complete and non-redundant description of the
triangle, so `g` is the transfer's full geometry rather than a projection of it.
`g = 0` at the reference triangle, so `V = A` there exactly.

**RMSD.** `ρ_m` is the optimally superposed RMSD (rotation and translation
removed, unit weights, no rescaling) to the `m`-th frame of the channel's stored
transition-state ensemble, averaged over the `M` frames.  An RMSD to a fixed
template is rotation- and translation-invariant but *not* scale-invariant, which
is why this form carries a nonzero virial rather than dropping out of the stress.
Under periodic boundaries the fragment is unwrapped by cumulative minimum image
before superposition, an affine operation in the cell.

### 6.1 Where the amplitude comes from

| form | amplitude from | width measured in |
| --- | --- | --- |
| two-body | the diabatic crossing | the breaking bond's length |
| three-body | the reference barrier | the transferring atom's triangle |
| RMSD | the reference barrier | the RMSD over all 3N coordinates |

For a channel with a true saddle, the amplitude is the reference barrier `E*`
inverted through the 2×2 secular equation at the transition state, where `V = A`
exactly:

```
A = -sqrt( (Hm - E*)^2 - dH^2 ),     Hm = (H_R + H_P) / 2
                                     dH = (H_R - H_P) / 2
```

A fission has no saddle and therefore no barrier to invert.  The bound diabat
*is* the ground state up to the crossing, so a perfect reactant diabat puts the
reference energy exactly on it, the discriminant vanishes identically, any
imperfection makes it imaginary, and the best attainable `A` is zero — a channel
that never opens.  A fission coupling is therefore centred on the diabatic
crossing instead.  This also makes the admission criterion self-consistent: at a
crossing the diabats are degenerate, so the stabilization `hypot(dH, V) - |dH|`
equals `|A|` exactly, and a Gaussian centred there is at its maximum precisely
where the topology decision is taken.

A crossing-centred fit is correspondingly wrong for a transfer: water
autoionization's two diabats never cross along the proton coordinate at all, the
products being 9.8 eV uphill in the gas phase, so there is no degeneracy to
centre on.

### 6.2 Why a transfer does not use an RMSD width

An RMSD is a tolerance on all `3N` coordinates at once, so any spectator atom
switches the coupling off.  With water autoionization's three transfer atoms held
exactly at the transition state, displacing only the three spectators by 0.1 Å —
less than thermal motion at 300 K — takes an RMSD coupling from 4.14 eV to
8.4e-3 eV, while the triangle form holds at 4.14 eV.  Under the RMSD form a
64-water box admits no diabatic state at any threshold: neutral water at 300 K
never puts all `3N` coordinates that close at once simultaneously.

The RMSD form is retained only as the fallback for a channel that is neither a
fission nor a transfer — two independent bond changes sharing no atom have no
single coordinate to be a function of.

---

## 7. Parameters

### 7.1 Per-term parameters

| term | parameters | units |
| --- | --- | --- |
| bond (Morse) | `D, r0, k, c, b` | kJ/mol, nm, kJ/mol/nm², —, — |
| angle | `theta0, k` | rad, kJ/mol |
| bond–bond | `r1_0, r2_0, k` | nm, nm, kJ/mol/nm² |
| bond–angle | `theta0, r0, k` | rad, nm, kJ/mol/nm |
| angle–angle | `theta1_0, theta2_0, k` | rad, rad, kJ/mol |
| periodic dihedral | `phi0, n, k` | rad, —, kJ/mol |
| dihedral–bond | `+ r0` | nm |
| dihedral–angle | `+ theta0` | rad |
| dihedral–angle–angle | `+ theta0_1, theta0_2` | rad |
| reference | `E0` | kJ/mol |
| electrostatics, per atom | `mu, eta, soft_amp, soft_decay` | see §4.2 |
| 12-6, per atom | `sigma, eps` | nm, kJ/mol |
| ZBL | none (atomic numbers only) | — |
| two-body coupling | `A, a, r0` | eV, 1/Å², Å |
| three-body coupling | `A, a, ra0, rb0, t0` | eV, 1/Å², Å, Å, rad |
| RMSD coupling | `A, a` + TS ensemble | eV, 1/Å² |

Exclusion terms carry no independent parameters: they reuse the 12-6 pair
parameters, the atomic numbers, and the charge kernel respectively.

### 7.2 Global parameters

These are properties of a *dataset*, not universal constants: a parameter set is
fitted at particular values and is only valid at those values.  The defaults
below apply where a dataset does not state a value.

| parameter | default | unit | enters |
| --- | --- | --- | --- |
| `bond_asymptote` | 1.0 | eV | Morse (§1.1) |
| `shape_decay` | 4.0 | — | Morse `b` fallback (§1.1) |
| `taper_radius` | 1.5 | Å | ZBL switch (§2) |
| `taper_width` | 0.12 | Å | ZBL switch (§2); sets `switch_width` by default |
| `switch_radius` | 0.22 | nm | 12-6 switch (§3) |
| `switch_width` | `taper_width / 10` | nm | 12-6 switch (§3) |
| `core_fraction` | 0.4 | — | 12-6 core continuation (§3) |
| `exclusion_depth` | 3 | bonds | exclusions (§5) |
| `exclude_coulomb` | true | — | exclusions (§5.2) |
| `gamma` | 2.0 | 1/Å | charge kernel (§4.1) |
| `accuracy` | 1e-8 | — | Ewald splitting and cutoff (§4.1) |
| `ccoul` | 14.4 | eV·Å | electrostatic energy (§4.3) |
| `zbl_ccoul` | 14.399645 | eV·Å | ZBL (§2) |

`switch_width` is the same physical width as `taper_width` expressed in the other
length unit — 0.12 Å is 0.012 nm.  `ccoul` and `zbl_ccoul` are the same physical
constant to different precision; they are separate fields so that unifying them
cannot silently move one of the two terms for an already-fitted parameter set.

Everything in the table except `accuracy` and `shape_decay` enters
`E_bonded + E_nonbonded`, which the bond depths are fitted against.  Changing any
of them invalidates the fitted parameters of a dataset.

### 7.3 Evidence behind the defaults

**`bond_asymptote` = 1.0 eV.** Chosen from the number of fittable reaction
channels and the resulting barrier margins, refitting at each candidate value:

| asymptote | fittable channels | rxn_16 | rxn_11 | rxn_15 |
| --- | --- | --- | --- | --- |
| 0.00 | 14 | −0.402 | +0.086 | +0.144 |
| 0.50 | 15 | −0.152 | +0.426 | +0.416 |
| 0.75 | 15 | −0.028 | +0.591 | +0.549 |
| 1.00 | 16 | +0.093 | +0.753 | +0.680 |
| 1.50 | 16 | +0.333 | +1.068 | +0.938 |
| 2.00 | 16 | +0.567 | +1.371 | +1.190 |

1.0 eV is the first value at which `rxn_16` — a genuine saddle, and the one
channel that was ever a fitting failure rather than a barrierless one — becomes
fittable; beyond 1.5 eV nothing further is gained.  The asymptote also sets where
a bonded diabat crosses its own fragments', which determines whether the reverse
channel is available where the forward one hands over:

| asymptote | H2 | HO | H2O |
| --- | --- | --- | --- |
| 0.75 | 2.53 Å | 3.15 Å | 4.05 Å |
| 1.00 | 2.38 Å | 2.93 Å | 3.74 Å |

**`shape_decay` = 4.0 (fallback only; `b` is fitted per bond type).** Refitting
the whole pipeline at each fixed `b`:

| b | c_max | metathesis channels | dissociation rms | fastest mode | stable dt |
| --- | --- | --- | --- | --- | --- |
| 2.0 | 1.31 | 12/13 | 0.700 eV | 4517 cm⁻¹ | 0.492 fs |
| 2.5 | 3.84 | 13/13 | 0.783 | 4402 | 0.505 |
| 3.0 | 7.80 | 13/13 | 1.098 | 4352 | 0.511 |
| 4.0 | 19.33 | 13/13 | 1.710 | 4400 | 0.505 |
| 5.0 | 34.50 | 13/13 | 2.660 | 4402 | 0.505 |
| 6.0 | 52.20 | 12/13 | 3.181 | 4432 | 0.502 |

The timestep is flat across the range — the curvature cap absorbs whatever `b`
does — so `b` costs nothing dynamically, and 4.0 is simply the worst reachable
value for the dissociation curves; hence it is fitted rather than fixed.
`c_max(b)` is the monotonicity limit on the shape amplitude, and it is why small
`b` is not free either: at `b = 2.0` it is 1.31 and binds on five of eight bond
types.

**`taper_radius` = 1.5 Å, bounded from both sides.** From below, the repulsive
wall must stay ahead of the electrostatic contact funnel at every separation: at
1.2 Å an H2 + O2 approach reads +0.52 eV against +3.14 eV untapered, and by 1.0 Å
that approach is downhill — i.e. the system collapses.  From above, every
additional 0.1 Å of reach costs roughly 0.2 eV of hydrogen-bond depth: the water
dimer minimum is −0.148 eV at 1.4 Å, −0.116 eV at 1.5 Å and −0.090 eV at 1.6 Å,
against a −0.218 eV reference at 2.91 Å.  1.5 Å places the minimum at the correct
*separation* and takes the depth deficit as a known residual.

**`taper_width` = 0.12 Å** is the smallest width that keeps the switch smooth
enough to integrate: its peak contribution to `du/dr` is 2.4 eV/Å at the O–H bond
length, well under the 17.8 eV/Å the unmodified term already carries there.

**`switch_radius` = 0.22 nm, deliberately not `taper_radius`.** The tidy design
`g = 1 - f` at the ZBL radius fails against real 12-6 parameters: a Fermi switch
decays by one factor of `e` per width while 12-6 grows as `r^-12`, so at 1.5 Å
the switch is down to 1.1e-2 where the bare 12-6 is up at 953 eV — a product of
+10.3 eV per O–H bond, or +20.6 eV on a single water molecule.  Bare 12-6 versus
what survives at 2.2 Å with a 0.012 nm width:

| pair | bare | switched |
| --- | --- | --- |
| O–H bond, 0.958 Å | 953 eV | 0.031 eV |
| H–H bond, 0.741 Å | 892 eV | 0.004 eV |
| O–O bond, 1.208 Å | 1375 eV | 0.353 eV |
| water 1-3 H···H, 1.51 Å | 0.138 eV | 0.0004 eV |
| hydrogen-bond O···H, 1.94 Å | 0.146 eV | 0.015 eV |
| O···O contact, 2.6 Å | 0.076 eV | 0.073 eV |
| O···O contact, 2.4 Å | 0.262 eV | 0.220 eV |

The last two rows are the intermolecular wall this term exists to supply,
retained at 97% and 84%.  Suppressing the term at bond lengths to 0.03–0.35 eV is
what allows it to carry no exclusions at all, hence to take the same value on
every state, hence to sit outside the Hamiltonian.  The resulting gap from ~1.6
to ~2.0 Å, where neither repulsive term acts (the ZBL taper is down to 0.076 at
1.8 Å and 12-6 is not yet on), is deliberate: electrostatics alone binds the
water dimer at −0.112 eV through it, and an intermolecular approach remains
uphill across the gap.

**`core_fraction` = 0.4.** `0.4 σ` is 0.78 Å for an H–H pair and 1.18 Å for O–O —
inside the Morse core of a real bond and far inside any intermolecular contact —
so the linear continuation never engages on a pair whose repulsion is doing
physical work; the wall at those radii is already 451 eV (H–H) and 1750 eV (O–O).

**`accuracy` = 1e-8.** Loosening it is cheap in reciprocal-vector count, which
scales as `(-log accuracy)^3`, but expensive in the *stress*: `kappa` is derived
from the cell, so a strained cell truncates at a slightly different splitting and
that residual is a spurious contribution to the numerical virial with no analytic
counterpart, amplified by `2 log(1/accuracy)` relative to the truncation error
itself.  At 1e-8 the residual is around 1e-6 eV.  Reciprocal vectors are selected
on an integer ellipsoid rather than by `|k|`, so the set is piecewise constant in
the cell and a strain cannot move a whole degenerate shell across the cutoff.
