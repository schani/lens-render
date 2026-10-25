# Explainer lenses

The lens sequence of the explainer video "Why does a sharp picture take six
pieces of glass?": 50 mm, 36 × 24 mm, focused at 2.5 m.

Each `NN-name.lens` is generated from `NN-name.toml` (start design,
variables, constraints, merit) by `lens-render optimize`
(`src/optimize.rs`). `scripts/explainer-lenses.sh` regenerates all lenses,
`bending.json` and `numbers.json` (the rows of the tables below, in order).
#7 and #11 are `../cooke-triplet.lens` and `../double-gauss.lens`, unchanged;
`07-triplet.lens` is #7 re-optimised for 2.5 m (the video uses it).

## Method

- **Optimiser**: damped least squares (Levenberg–Marquardt), finite-difference
  Jacobian, random restarts around the best design. Merit: white-light RMS
  spot (F, e, C weighted 1:2:1) from real rays at 0, 0.5, 0.7, 1.0 × 21.63 mm,
  object at 2.5 m, on the sensor plane the renderer focuses to (axial
  least-squares blur). Penalties: glass centre ≥ 1.5 mm, edge ≥ 1 mm, air
  ≥ 0.1 mm, back focus ≥ 15 mm, vignetting limit. EFL held at 50 mm by
  scaling all curvatures. Clear apertures are variables; the job's
  `vignetting` sets how much light the corner may lose (0.2 for #5/#6, 0.5
  for #8–#11b; the singlets and #4 may not vignette).
- **Numbers** (`lens-render numbers <lens>@<f> --focus 2.5`), lens focused
  as the renderer does:
  - RMS: white-light (F, e, C 1:2:1) RMS spot radius with all apertures, at
    0, 0.7 and 1.0 of the half-diagonal.
  - Distortion at the corner: real chief ray vs. paraxial image height
    (negative = barrel). C–F: paraxial focus of C minus F. Petzval radius
    −1/Σ c Δn/(n n') (negative = curved towards the lens). Corner
    illumination: cos⁴, vignetting and distortion, no reflection losses.
  - Seidel sums (`seidel::seidel`, full field, waves at 550 nm) and their
    transverse sizes on the sensor (`seidel::transverse_um`): the image-plane
    error of the pupil-edge ray, ε = 2N′ ∂W/∂ρ, with N′ the image-side working
    f-number: spherical 8N′W040, coma (tangential) 6N′W131, astigmatism
    4N′W222, field curvature (Petzval) 4N′W220P, distortion 2N′W311, axial
    colour 4N′W020, lateral colour 2N′W111, × λ. Positive W311 is barrel.

## Numbers

| # | f/ | EFL mm | BFD mm | length mm | RMS centre µm | RMS 0.7 µm | RMS corner µm | distortion % | C–F focus µm | Petzval R mm | corner illum. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4.00 | 50.00 | 49.2 | 52.2 | 36.9 | 594.1 | 1227.3 | -0.97 | 798 | -75 | 0.73 |
| 2 | 4.00 | 50.00 | 48.9 | 51.9 | 30.6 | 593.6 | 1241.7 | -0.66 | 797 | -76 | 0.73 |
| 2b | 4.00 | 50.00 | 48.5 | 51.5 | 79.4 | 678.6 | 1355.2 | -1.46 | 806 | -76 | 0.73 |
| 3 (2 at f/16) | 16.00 | 50.00 | 49.4 | 52.4 | 6.2 | 143.3 | 300.2 | -0.64 | 797 | -76 | 0.73 |
| 4 | 8.00 | 50.00 | 48.2 | 54.7 | 1.0 | 276.7 | 581.5 | -1.29 | 45 | -69 | 0.75 |
| 5 | 14.00 | 50.00 | 53.2 | 67.1 | 3.2 | 19.1 | 34.3 | -3.62 | 0 | -76 | 0.59 |
| 6 | 8.00 | 50.00 | 40.4 | 61.7 | 10.6 | 34.9 | 61.7 | +0.40 | 347 | -75 | 0.57 |
| 7 | 3.50 | 50.02 | 43.2 | 61.2 | 5.1 | 49.9 | 55.2 | +0.17 | 87 | -128 | 0.61 |
| 07 (7 re-opt.) | 3.50 | 50.00 | 42.0 | 59.5 | 5.2 | 24.0 | 11.5 | -1.15 | 105 | -116 | 0.31 |
| 8 | 2.00 | 50.00 | 35.2 | 60.3 | 34.5 | 54.3 | 31.3 | +0.76 | 199 | -101 | 0.36 |
| 9 | 4.00 | 50.00 | 39.8 | 59.4 | 16.6 | 32.3 | 31.0 | +1.02 | 421 | -281 | 0.44 |
| 10 | 2.80 | 50.00 | 37.5 | 61.6 | 20.2 | 72.9 | 73.3 | +1.75 | -1 | -141 | 0.36 |
| 11 | 2.03 | 50.36 | 37.1 | 69.1 | 19.6 | 36.4 | 48.1 | -1.74 | 206 | -316 | 0.17 |
| 11b | 2.00 | 50.00 | 31.1 | 62.8 | 16.7 | 18.0 | 21.4 | +0.93 | 24 | -172 | 0.35 |

Seidel sums, waves / transverse µm (the scoreboard bars):

| # | N′ | spherical | coma | astigmatism | field curv. | distortion | axial colour | lateral colour |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.00 | 8.55 / 151 | -43.71 / 577 | 118.57 / 1044 | 44.14 / 389 | 45.98 / 202 | 10.96 / 96 | 2.43 / 11 |
| 2 | 4.02 | 5.63 / 100 | -3.63 / 48 | 126.46 / 1120 | 43.46 / 385 | 30.99 / 137 | 10.94 / 97 | 1.84 / 8 |
| 2b | 3.89 | 23.39 / 400 | -99.78 / 1280 | 108.04 / 924 | 46.39 / 397 | 70.88 / 303 | 11.09 / 95 | 3.37 / 14 |
| 3 (2 at f/16) | 16.27 | 0.02 / 2 | -0.06 / 3 | 7.74 / 277 | 2.66 / 95 | 7.51 / 134 | 0.68 / 24 | 0.45 / 8 |
| 4 | 8.16 | -0.07 / 3 | 0.01 / 0 | 28.94 / 520 | 11.54 / 207 | 30.28 / 272 | 0.15 / 3 | 0.11 / 1 |
| 5 | 14.13 | 0.31 / 19 | -0.38 / 18 | -2.93 / 91 | 3.52 / 109 | 50.56 / 786 | -0.00 / 0 | 1.58 / 25 |
| 6 | 8.06 | 1.49 / 53 | 2.49 / 66 | -7.84 / 139 | 10.89 / 193 | -8.66 / 77 | 1.18 / 21 | 0.48 / 4 |
| 7 | 3.55 | 6.81 / 107 | -8.26 / 97 | -21.55 / 169 | 32.83 / 257 | 0.47 / 2 | 1.48 / 12 | 0.05 / 0 |
| 8 | 2.00 | 68.48 / 603 | -116.46 / 769 | 4.58 / 20 | 131.70 / 580 | -189.43 / 417 | 10.63 / 47 | -21.53 / 47 |
| 9 | 4.06 | 3.51 / 63 | 16.12 / 216 | 1.61 / 14 | 11.50 / 103 | -47.16 / 211 | 5.72 / 51 | -1.70 / 8 |
| 10 | 2.83 | 12.75 / 159 | 43.11 / 402 | -0.07 / 0 | 47.18 / 294 | -108.98 / 339 | -0.08 / 1 | -0.90 / 3 |
| 11 | 2.07 | 10.28 / 93 | -9.36 / 64 | -17.32 / 79 | 39.42 / 179 | 128.24 / 292 | 10.75 / 49 | 7.43 / 17 |
| 11b | 2.03 | 24.16 / 215 | -6.97 / 47 | -1.33 / 6 | 75.15 / 335 | -131.19 / 292 | 1.13 / 5 | -2.55 / 6 |

Speed (exposure ∝ N²): triplet f/3.5 vs Rapid Rectilinear f/8: 5.2×;
double Gauss f/2 vs RR f/8: 16×; RR f/8 vs landscape f/14: 3.1×; f/4 →
f/16: 16× (1/125 → 1/8 s).

## Bending (`bending.json`)

Singlet of #1 bent at constant thickness and EFL, shape factor
X = (c1 + c2)/(c1 − c2) from −1 (flat front) to +2 in steps of 0.1, f/4,
2.5 m: radii (0 = flat), W040, white-light centre RMS (renderer focus) and
546 nm RMS (own best focus, spherical aberration alone). Best form:
X = 0.70 (R1 30.3, R2 −169 mm). Equiconvex / best form: W040 ×1.52, 546 nm
RMS ×1.53 (26.9 → 17.6 µm). Flat front / best: ×4.2. In white light the
centre RMS only drops 36.9 → 30.6 µm (×1.2): axial colour dominates.

## The designs

1. **Equiconvex singlet, f/4.** Glow (W040 8.6 waves), colour (C–F 0.8 mm),
   corners smeared over ~1 mm (astigmatism 119 waves). As claimed.
2. **Best form (X = 0.70), f/4; 2b flat front.** Spherical aberration ×1.5
   lower than equiconvex, ×4.2 lower than flat front; coma nearly gone
   (W131 −3.6). Rows match #1 and 2b.
3. **#2 at f/16.** Centre RMS 6.2 µm, below the Airy radius (11 µm) and a
   pixel; corners still ~0.3 mm.
4. **Cemented achromat, f/8.** Centre 1.0 µm, C–F 45 µm (×18 less than the
   singlet), coma 0; corners 0.6 mm (astigmatism, field curvature). As claimed.
5. **Landscape lens, f/14** (stop 9.4 mm in front). f/14 is the fastest
   f-number in f/11–f/16 where the 0.7-field RMS stays below one pixel of a
   1920-px frame (19 µm; scan: f/11 22, f/12.5 20, f/14 18 µm). Corner 34 µm, barrel
   distortion −3.6 % at the corner (0.8 mm, clearly bowed grid). As claimed.
6. **Rapid Rectilinear, f/8.** Distortion +0.4 % (9× less than #5),
   lateral colour 4 µm; centre 11 µm, corner 62 µm (Petzval −75 mm). At f/14
   it is no sharper than #5 (3.5 / 18 / 38 µm): its gains are speed and
   straight lines.
7. **Cooke triplet, f/3.5** (as is). 5 / 50 / 55 µm. For 2.5 m it is not
   optimal by this merit: re-optimised (same glasses, `07-triplet.lens`) it
   reaches 5 / 24 / 11 µm, with −1.2 % distortion and 31 % corner
   illumination. File unchanged.
8. **Triplet at f/2** (same glasses, everything free, 30 restarts).
   35 / 54 / 31 µm; on axis alone it could reach 14 µm, but not together
   with the field (0 + 0.5 field: 30 / 35 µm).
9. **Clark type, f/4.** 17 / 32 / 31 µm but C–F 421 µm (purple fringe on the
   centre light). At f/2.8 the best found was 52 / 63 / 60 µm (C–F 730 µm).
10. **Planar type, f/2.8** (N-SK16/F2, equal index, buried surface). Colour
    corrected (C–F −1 µm); centre 20 µm, field 73 µm (coma 43 waves).
11. **Double Gauss (as is), max f/2.03.** 20 / 36 / 48 µm with 17 % corner
    illumination; C–F 206 µm (its Abbe numbers are assumed, see the file).
    **11b**, the same layout re-optimised with N-LAK9/N-SF5: f/2,
    17 / 18 / 21 µm, C–F 24 µm, coma −7 waves, corner illumination 35 %.

## Treatment claims the numbers contradict

1. **"Even now [f/16], the centre isn't clean."** It is: RMS 6.2 µm, below
   the diffraction blur; no visible fringe in a ×4 inset. Say instead: "At
   f/16 the glow is gone, but so is most of the light", and show the colour
   with the by-wavelength fans or the focus zoom (C–F 0.8 mm).
2. **"Spherical aberration drops [×N]".** N = 1.5 from equiconvex (×4.2 from
   flat front). The white-light centre blur only improves ×1.2 because
   colour dominates; show the drop in the focus zoom or at one wavelength.
3. **Scoreboard "green below 15 µm", "all bars green" at the double Gauss.**
   Seidel bars at full aperture and field are tens to hundreds of µm for
   every lens here, also the good ones (triplet: coma 97, Petzval 257 µm;
   11b: spherical 215, Petzval 335, distortion 292 µm). Good lenses balance
   third order against higher orders and terms against each other. Say "the
   bars are big but they cancel", or base the bars on real-ray measures, and
   let the measured RMS readouts carry the verdict.
4. **f/2 triplet "bars tame while measured blur is high".** Opposite: the
   bars explode (spherical 603, coma 769, Petzval 580 µm) while the measured
   blur is 35 / 54 / 31 µm. Say "the third-order terms grow huge and only
   cancel in part".
5. **Symmetry cancels coma (RR, Clark, Planar).** Only near 1:1. At 2.5 m the
   residual coma is 2.5 waves (RR), 16 (Clark), 43 (Planar); distortion and
   lateral colour do mostly cancel. The double Gauss's asymmetry is what
   removes the coma (11b: −7 waves). Say "symmetry cancels distortion and
   lateral colour; coma cancels only for close subjects; tuning the halves
   separately removes the rest" (chapter 16 then has a concrete payoff).
6. **"Sharp to the corners at f/3.5" (Cooke triplet).** Corners 55 µm
   (~3 px). Say "sharp in the centre, usable corners, at f/3.5".
7. **Clark (#9) as a fix after the f/2 triplet.** It is not one: at f/2.8 it
   is worse than the f/2 triplet everywhere (52 / 63 / 60 µm); at f/4 its
   corners beat the Cooke triplet's (31 vs 55 µm) but its centre is 3×
   softer, with 5× the axial colour. Present it as the idea that still
   needed Rudolph's step, not as an improvement.
8. **Planar: "the thick glass flattens the field".** Its Petzval radius
   (−141 mm) is not flatter than the Clark's (−281 mm); what it fixes is
   colour (C–F 421 → −1 µm). Planar vs Clark at the same f-number: at f/2.8
   the Planar wins in the centre (20 vs 52 µm) and colour, the field is
   similar or worse (73 vs 60 µm); both optimised at f/4: Planar 7–13 /
   34–39 / 32 µm vs Clark 17 / 32 / 31 µm. Say "the buried surface fixes
   colour"; drop the field-flattening claim.
9. **Double Gauss f/2 (`double-gauss.lens`).** `fstop = 2` fails: the stop
   allows f/2.03. EFL 50.36 mm (outside ±0.5 %). Centre 20 µm with a magenta
   ring (C–F focus shift 2.4× the triplet's), corners rely on 17 % illumination.
   Suggest using `11b-double-gauss.lens` (f/2, 17 / 18 / 21 µm). "Sixteen
   times faster than the Rapid Rectilinear" holds (N² 64/4).
