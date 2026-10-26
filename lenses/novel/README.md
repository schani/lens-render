# Few-element 50 mm f/2 lenses with off-map materials

Question: does a material outside the glass map, above all diamond
(n_d 2.42, V 55), let ≤ 3 elements reach a flat, sharp full-frame field at
f/2, which glass cannot? Exotic materials count only if the result is
spectacular.

**Verdict: no.** Nothing reached the bar with ≤ 3 elements. Diamond makes a
3-element f/2 about 1.5× sharper off axis than glass, but the cubic garnets
LuAG and YAG with a polystyrene flint do as well, and the diamond would have
to be 25 mm across. Cut to the 20 mm single-crystal diamond actually sold,
the diamond triplet is worse than the glass triplet. The structure that
wins is a Cooke triplet with aspheres, a known type.

## Bar

50 mm, 36 × 24 mm (half-diagonal 21.63 mm), focus at infinity, ≤ 3
elements, f/2 or faster. White-light RMS spot ≤ ~8 µm at 0, 0.7 and 1.0
field. Distortion ≤ 1 %, corner illumination ≥ 0.35, back focus ≥ 20 mm,
length ≤ 90 mm. Glass aspheres depart ≤ 0.5 mm from the sphere. Diamond
surfaces spherical (or ≤ 0.05 mm), diamond ≤ 6 mm thick. Glass centre
≥ 1.5 mm, edge ≥ 1 mm.

## Result (f/2, infinity, `lens-render numbers --fields`)

RMS: F, e, C 1:2:1, all apertures, sensor where the renderer focuses.

| Lens | Elements | RMS 0 / 0.7 / 1.0 µm | 440–660 nm flat | distortion | corner illum. | length / back focus mm | at 2.5 m |
|---|---|---|---|---|---|---|---|
| `diamond-triplet-f2` (D) | diamond · N-SF11 · LuAG | 6.7 / 9.1 / 9.9 | 6.5 / 8.8 / 10.9 | +1.00 % | 0.36 | 61.3 / 20.7 | 7.7 / 9.8 / 12.3 |
| `garnet-triplet-f2` (X) | LuAG · polystyrene · YAG | 5.0 / 10.0 / 9.0 | 5.8 / 9.5 / 10.3 | −0.29 % | 0.39 | 75.3 / 42.1 | 5.9 / 10.7 / 12.3 |
| `glass-triplet-f2` (G) | S-LAH65V · N-SF11 · S-LAH55V | 7.5 / 14.4 / 14.8 | 8.3 / 13.5 / 16.2 | +0.99 % | 0.38 | 55.7 / 29.6 | 8.4 / 15.7 / 17.0 |
| `diamond-triplet-20mm-f2` (D, diamond ≤ 20 mm) | LuAG · N-LASF46B · diamond | 10.8 / 15.1 / 17.6 | 11.7 / 14.6 / 19.1 | −0.73 % | 0.35 | 72.6 / 39.5 | 12.0 / 16.4 / 14.8 |
| `../explainer/11b-double-gauss.lens` | 6, spherical | 14.7 / 20.5 / 22.3 | 16.7 / 19.6 / 23.4 | +1.16 % | 0.32 | 61.9 / 30.1 | 16.7 / 18.0 / 21.4 |
| `../summilux-50-asph.lens` at f/1.4 | 8, one asphere | 2.4 / 8.7 / 34.7 | 3.1 / 9.0 / 35.0 | +0.60 % | 0.21 | 76.4 / 25.2 | 2.5 / 13.6 / 32.1 |
| `../summilux-50-asph.lens` at f/2 | | 2.5 / 8.5 / 29.5 | | +0.60 % | 0.40 | | 2.5 / 13.5 / 26.4 |

All four new lenses: every field 0, 0.3, 0.5, 0.7, 0.85, 1.0 checked (no
hidden bump), no ray failures, no crossing surfaces, edges ≥ 1 mm, aspheric
departures ≤ 0.18 mm, the diamond's surfaces spherical (diamond triplet) or
≤ 0.05 mm (20 mm variant). Flux at the corner is 0.34–0.39 of the centre; no
pupil collapse. Lateral colour 440–660 nm at the corner: diamond +7.7 µm,
garnet −1.8, glass +23.4 (visible as a green–magenta rim on the corner
light), double Gauss +1.8 with a purple axial halo.

Geometric MTF tells the same story (wave optics, white light, f/2,
sagittal / tangential at 40 lp/mm): centre 0.52 diamond, 0.77 garnet,
0.54 glass, 0.06 double Gauss; at 15 mm image height the sagittal
40 lp/mm drops to 0.06 (diamond), 0.30 (garnet), 0.15 (glass), 0.03
(double Gauss).

Nothing met the f/2 bar, so f/1.8, f/1.6 and f/1.4 were not run.

## Protocol

- Material sets. **G**: glasses (N-PK52A, CaF₂, N-LAK9, N-LAK34,
  N-LAK33B, S-LAH65V, S-LAH55V, S-LAH58, S-LAH79; F2, N-SF5, N-SF11,
  N-KZFS4, N-LASF46B). **X**: everything but diamond: G plus YAG, LuAG,
  spinel, BaF₂, LiF, PMMA, polycarbonate, polystyrene, cubic zirconia, ZnS.
  **D**: X plus diamond. Positive slots offered 13 materials (5 dropped
  after the first screening: LiF, PMMA, S-LAH79, never near the top; ZnS
  and N-KZFS4 from the negative slots), negative slots 7.
- Topologies. T1 cemented doublet (stop front / rear, flint first). T2
  air-spaced doublet (stop front, between, negative first). T3A Cooke
  triplet, stop behind the negative; T3B Cooke triplet, stop in front of
  the negative; T3C cemented doublet, stop, singlet; T3D singlet, stop,
  cemented doublet; T3E two positives and a rear negative, front stop.
- Merit (identical for all): RMS at 0, 0.3, 0.5, 0.7, 0.85, 1.0 field,
  440/486/546/656 nm weighted 0.5:1:2:1, object at 1000 m; vignetting up to
  half the light at the corner, distortion ≤ 1 %, corner illumination
  ≥ 0.36, length ≤ 90 mm, back focus ≥ 20 mm, departure ≤ 0.5 mm, slope
  ≤ 0.1, ≤ 2 inflections. Diamond: ≤ 6 mm thick, departure ≤ 0.05 mm
  (`[[material_limits]]`).
- Search, same budget and seeds for every set:
  1. spherical screening, one start per glass combination (≈ 1 200 per
     triplet, all sets at once);
  2. aspheric refinement of the best 2 combinations per set (12 starts,
     one basin hop each);
  3. final: 24 starts with one hop from the best 3 per set (9 runs);
  4. the committed jobs: 24 more starts from each set's best.
- Every finalist checked with `lens-render numbers --fields
  0,0.3,0.5,0.7,0.85,1` at infinity and 2.5 m, at F/e/C and over 440–660 nm
  (flat and luminance-weighted).

## Experiment table

RMS 0 / 0.7 / 1.0 µm at f/2, infinity; (materials). "Screen": spherical,
best of one start per combination, merit values. "Asph.": aspheric, verified
with `numbers` unless marked (m) for merit values.

| Topology | G | X | D |
|---|---|---|---|
| T1A cemented, stop front, asph. (m) | 31 / 128 / 236 (N-PK52A/F2) | – | same; diamond not picked |
| T1B cemented, stop rear, asph. (m) | 74 / 161 / 251 | – | 90 / 140 / 244 (CaF₂/N-KZFS8) |
| T1C cemented, flint first, asph. (m) | 51 / 121 / 186 | – | 46 / 122 / 200 (F2/CaF₂) |
| T2A air-spaced, stop front, screen | 164 / 179 / 182 | 149 / 163 / 177 | 257 / 260 / 205 (diamond/CZ) |
| T2B air-spaced, stop between, screen | 152 / 149 / 186 | 140 / 144 / 157 | 120 / 133 / 173 (diamond/CZ) |
| T2B asph. (m) | 82 / 83 / 70 | 96 / 89 / 68 | 72 / 80 / 85 |
| T2C negative first, screen | 153 / 162 / 185 | 133 / 139 / 158 (CZ/YAG) | 136 / 147 / 150 (N-SF5/diamond) |
| T3A triplet, stop behind negative, screen | 20 / 28 / 27 | 18 / 24 / 26 (YAG/PC/YAG) | 12 / 19 / 22 (LuAG/N-LASF46B/diamond) |
| T3A asph. | 9.7 / 18.5 / 14.0 (S-LAH65V/N-SF11/S-LAH58) | 9.4 / 18.5 / 17.7 (YAG/PC/YAG) | 9.2 / 17.2 / 17.6 (LuAG/N-LASF46B/diamond) |
| T3B triplet, stop in front of negative, screen | 20 / 28 / 26 | 18 / 25 / 24 (LuAG/PS/YAG) | 13 / 18 / 20 (diamond/PC/LuAG) |
| **T3B asph.** | **7.5 / 14.4 / 14.8** | **5.0 / 10.0 / 9.0** (LuAG/PS/YAG) | **6.7 / 9.1 / 9.9** (diamond spherical/N-SF11/LuAG) |
| T3B asph., no polymer | – | 6.4 / 19.1 / 16.3 (LuAG/N-SF11/YAG) | 5.3 / 9.9 / 10.6 (diamond aspheric/N-SF11/LuAG) |
| T3B asph., polycarbonate | – | – | 5.9 / 9.0 / 11.1 (diamond/PC/LuAG) |
| T3C cemented + singlet, screen | 70 / 98 / 157 | = G | 78 / 77 / 81 (diamond/PC/diamond) |
| T3C asph. | 18 / 37 / 31 | = G | 10 / 23 / 30 (m) |
| T3D singlet + cemented, screen | 64 / 92 / 143 | = G | 70 / 74 / 75 (LuAG/PS/diamond) |
| T3D asph. | 20 / 41 / 52 | = G | 10 / 13 / 20 (LuAG · PS+diamond) |
| T3E front stop, rear negative, screen | 117 / 111 / 134 | 62 / 84 / 86 | 67 / 70 / 82 |
| T3E asph. | 35 / 41 / 38 | 34 / 45 / 48 | 41 / 41 / 45 (m) |

T1, T2, T3C–E and the screening ran before the merit fixes below and with
the earlier polymer data; their errors are a few µm, small against their
distance from the bar.

## Which material earns its place

- **Diamond**: wins the spherical screening of every triplet (the 8 best
  T3A and the 38 best T3B combinations all contain it); with aspheres on the glass
  elements its lead over glass is 1.1 / 1.6 / 1.5× at 0 / 0.7 / 1.0 field,
  and it needs no asphere itself (spherical and aspheric diamond: 6.7 / 9.1
  / 9.9 vs 5.3 / 9.9 / 10.6 µm).
- **LuAG** (n 1.84, V 58) and **YAG** (1.83, 53): in every good X and D
  design. They sit ~10 Abbe units above the glass line (S-LAH55V: 1.83,
  43). Paired with a low-index flint they match diamond once glass aspheres
  are allowed. Spinel (1.72, 61) reached the top of the first screening
  but none of the finals.
- **Polystyrene / polycarbonate**: the X winner depends on polystyrene's
  low index (1.59) as the negative; with N-SF11 instead the garnet triplet
  falls to 6.4 / 19.1 / 16.3 µm. Their dispersion data are weak: the
  Sultanova 2009 one-term fits imply strongly anomalous partial dispersion
  (ΔP_g,F −0.023), the Zhang 2020 tabulated data −0.008 to −0.014, with
  ±0.02 noise. The catalogue now uses fits to Zhang's data; with
  Sultanova's, the garnet triplet as built gives 6.8 / 11.5 / 10.6 µm. A
  polystyrene element also brings dn/dT of order −10⁻⁴/K (about 100× glass)
  and moulding birefringence.
- **Not useful here**: CaF₂, BaF₂, LiF, N-PK52A (low index raises the
  Petzval sum and curves; secondary spectrum is not what limits a 3-element
  f/2), PMMA, cubic zirconia (only in the hopeless doublets), ZnS (too
  dispersive).
- ALON (n ≈ 1.79) and Murata Lumicera (n_d 2.08) are not in the catalogue:
  no dispersion data with a verifiable source were found (the Sellmeier
  remembered for ALON gave 1.786 at 633 nm against Surmet's measured 1.792).
  Zeonex E48R: only a one-term fit whose V (52) disagrees with the
  datasheet value, so left out.

## Diamond: size, feasibility, coating

- The best diamond element is 24.6 mm across, 4.0 mm thick, at the front.
  At f/2 the entrance pupil is 25 mm, so any front diamond is ≥ 25 mm.
- Single-crystal CVD diamond for optics comes in plates of roughly 10–20 mm.
  Limited to 20 mm (`max_semi_diameter = 9.9`), the search moves the diamond
  behind the stop, where the beam is narrowest (19.7 mm across); a
  positive LuAG element in front magnifies the pupil (stop 19.6 mm for the
  25 mm entrance pupil). Result 10.8 / 15.1 / 17.6 µm: worse than the glass
  triplet (7.5 / 14.4 / 14.8). The size limit costs 1.6× at the edge.
- Larger single crystals exist: heteroepitaxial diamond on Ir/YSZ/Si,
  ~90 mm across, 1.6 mm thick, 155 ct (Schreck, Gsell, Brescia & Fischer,
  "Ion bombardment induced buried lateral growth: the key mechanism for the
  synthesis of single crystal diamond wafers", Sci. Rep. 7, 44462 (2017)).
  Its dislocation density, ~4 × 10⁷ cm⁻² after 1.5 mm of growth, is high
  for optics: dislocation bundles cause stress birefringence, and the
  plate is thinner than the 4 mm element. Optical quality at that size is
  not established.
- Reflection per surface at normal incidence, 440–660 nm mean: uncoated
  17.2 % (garnets 8.7 %, S-LAH58 9.4 %); single MgF₂ layer 2.2 % (0.4 %);
  the renderer's multi-layer, whose tailored inner layer clamps at n 2.0,
  0.59 % (an ideal 2.15 layer: 0.48 %; garnets 0.24 %). With the default
  single coating the night render shows a large faint purple ghost from
  the diamond surfaces; with `coating=multi` it is gone.

## Ideas tried that failed (f/2, infinity, RMS 0 / 0.7 / 1.0 µm, merit)

1. Petzval-free cemented achromat: diamond + F2 has a Petzval sum of
   0.02 φ at powers 2.9 φ / −1.9 φ (N-LAK9 + F2: 0.54 φ). As a doublet with
   a front stop and an aspheric glass corrector behind it: 59 / 73 / 78.
   Zero Petzval does not flatten the field: the astigmatism of a strong
   doublet remains.
2. Stop inside a split diamond/flint doublet behind a glass element:
   32 / 35 / 70.
3. Thin strong diamond meniscus right behind the stop between glass
   positives: 45 / 54 / 52.
4. Diamond as the negative middle element of a Cooke triplet: 100 / 136 /
   124 (a high-index negative adds Petzval sum).
5. Diamond in a cemented doublet with polystyrene behind the stop (T3D):
   10 / 13 / 20; with glass only 20 / 41 / 52. Diamond helps a lot here,
   but the Cooke triplet is better.
6. Diamond field flattener near the sensor: not possible with ≥ 20 mm back
   focus. Monocentric designs need a curved sensor. The spectral annulus
   stop was not tried: colour is not the limit (C–F 100 µm is balanced
   against the field).

## Optimiser artefacts found and fixed

- The merit focused on the axial bundle over the *paraxial* pupil, unclipped;
  the renderer and `numbers` focus with the real pupil. For fast aspheric
  designs the planes differed: a diamond design showed merit 7.3 / 10.7 /
  11.6 against 5.1 / 13.6 / 10.8 µm verified, off-axis up to 25 % optimistic. Now
  the merit focuses with the real, clipped pupil (test:
  `render_focus_matches_best_focus`).
- Distortion and corner illumination in the merit used chief rays on the
  start design's image plane, not the focus: a design with merit distortion
  0.6 % had 2.0 % (test: `illumination_and_distortion_match_numbers` with a
  start image distance 12 mm off).
- Objects ≥ 1 km away: ray–sphere intersections lost precision (quadratic
  with 10⁶ mm origins) and chief-ray aiming did not converge, so `numbers`
  gave NaN at infinity. Rays now start just before each surface (test:
  `distant_object`).
- Polymer dispersion (above): the first X result (4.8 / 11.2 / 10.4 µm)
  relied on Sultanova's anomalous partial dispersion; with Zhang's data the
  same lens gave 11.3 / 13.6 / 12.1. All finals were redone on Zhang's data.

## Known type?

All finalists are Cooke triplets (+ − +) with the stop between the front
positive and the negative, every glass surface aspheric. Aspheric triplets
are standard in compact and phone cameras. The new part is the materials:
a spherical diamond crown, or LuAG/YAG garnet crowns with a polystyrene
flint (no prior art search was done for garnet camera lenses). Not a
landscape, Petzval, Tessar, Sonnar, Gauss, Hologon or monocentric lens.

## Files

- `diamond-triplet-f2`, `garnet-triplet-f2`, `glass-triplet-f2`,
  `diamond-triplet-20mm-f2`: `.toml` job (start = the previous stage's
  best) and the `.lens` it writes (`lens-render optimize
  lenses/novel/<name>.toml`, ~4–5 min each on 12 cores).
- New tooling: `lens-render numbers --fields 0,0.3,…` (RMS per field, also
  over 440–660 nm, lateral colour, distortion, flux); `[[material_limits]]`
  in jobs (`docs/optimiser.md`); catalogue: YAG, LuAG, spinel, BaF₂, LiF,
  PMMA, polycarbonate, polystyrene, N-LAK34, N-LAK33B, S-LAH65V, S-LAH55V
  (`src/glass.rs`, sources there).
