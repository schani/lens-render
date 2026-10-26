# Leica Summilux-M 50 f/1.4 ASPH — research notes

## Bottom line

- **There is no published numerical prescription.** The patent for this lens has no data table.
- **The 2023 lens uses the same optical formula as the 2004 Karbe design.** Both Leica lens-cut drawings give the same radii and spacings. They also show the same asphere profile.
- `summilux-50-asph.lens` is therefore a **reconstruction**. Its geometry is fitted to Leica's own vector lens drawing. Its glasses are refit to Leica's 2023 MTF (catalogue types plus two nd/vd fits; see "MTF refit"), not known ones. It reproduces Leica's focal length, entrance pupil, distortion, vignetting, close-focus magnification and the patent's partial focal lengths. Its MTF meets or beats Leica's at most points, with gaps at mid-field 40 lp/mm.

## Sources

| What | URL / file |
|---|---|
| Patent (matching lens) | DE 10 2004 008 997 B4: https://patents.google.com/patent/DE102004008997B4/de |
| US family member | US 7,102,834 B2: https://patents.google.com/patent/US7102834B2/en |
| JP family member | JP 2005-234564 A: https://patents.google.com/patent/JP2005234564A/en |
| 2023 data sheet (Feb 2023) | https://leica-camera.com/sites/default/files/pm-14169-EN_Datenblatt-Summilux-M-50---Classic.pdf |
| 2023 spec page | https://leica-camera.com/en-int/photography/lenses/m/leica-summilux-m-50-f14-asph-black/technical-specification |
| 2023 MTF images | https://leica-camera.com/sites/default/files/2025-05/11728_Leica_Summilux-M-50_mtf0{1,2,3}.jpg |
| 2004 brochure / data sheet | https://www.summilux.net/m_system/images/Summilux50Asph.pdf |
| "Same formula" opinion | diglloyd: https://diglloyd.com/blog/2023/20230413_1600-LeicaM50f1_4_II.html |
| "Slightly modified" opinion | Macfilos: https://www.macfilos.com/2023/04/26/review-leica-50-summilux-asph-2023-close-focus/ |
| Not reachable | Zhihu article guessing glasses/simulating the lens: https://zhuanlan.zhihu.com/p/1990194808648639474 (blocked, 403/40362). It is guesswork, not data. |

### Searches that found nothing usable

- Google Patents, all Peter Karbe inventions. The only relevant one is DE102004008997; DE10354716 / JP2004310109 cover a close-focus adapter mechanism with no optics data.
- Leica Camera AG optical patents filed after 2015. None is a 50/1.4 M lens.
- Bill Claff's Optical Bench (photonstophotos.net). It covers several Leica lenses but not the Summilux-M 50 ASPH.
- Japanese and Chinese searches and Marco Cavina. Nothing found.

## Patent identification

DE102004008997B4 / US7102834B2 (Peter Karbe, Leica Camera AG, priority 22 Feb 2004), "Fotografisches Objektiv vom modifizierten Doppelgaußtyp". Its second example (Fig. 2/2a, f = 50 mm, 1:1.4) matches the lens on every point:

- 8 elements in 5 groups: L1 | L2/L3 | stop | L4 | L5'/L5 | L6/L7.
- One asphere, on the stop-side concave surface of L4. This is Leica's "fourth element" asphere, at the same place in Leica's 2004 drawing.
- Floating element: the rear doublet L6/L7 moves more slowly than the rest. This matches Leica 2004, where elements 7/8 are the "floating element".
- Glasses: APD glass in L2/L3; n > 1.80 in L1, L5 and L6, which are Leica elements 1, 6 and 7.
- Close focus 0.70 m.

The patent gives only partial focal lengths (±5 %): f(L1) = 77, f(L2/L3) = −97, f(L4) = −369, f(L5'/L5) = 68, f(L6/L7) = 154 mm. It gives no radii, spacings, glasses or aspheric coefficients. The Fig. 1 example is the APO-Summicron-M 75 f/2 type. 

## 2004 vs 2023

- Leica's 2023 press text says "refined technically and optically". Its changes: 0.45 m close focus via a double cam (rangefinder coupling still to 0.7 m), 11 instead of 9 blades, and a new mechanical design.
- Both Leica lens cuts are vector drawings, so I fitted them. Scaled to the same S1-to-image length, all 13 radii agree within drawing precision, mostly < 1 %. R2 and R7 differ by 5 % but are weak surfaces.
  - The back focus agrees to 0.01 %.
  - The aspheric profile of L4 agrees: vertex R −449 vs −453 and A4 −2.91e-7 vs −2.95e-7 in the same units.
  - The only visible difference is the split of the L2/L3 doublet thickness, 29.1/9.2 vs 27.6/11.0 pt, with the total equal. This could be a drawing difference or a minor re-optimisation.
- The 2023 MTF curves have the same shape as 2004 but are higher on axis (40 lp/mm at f/1.4: 56 → 64 %; at f/2.8: 64 → 76 %). See `leica-mtf.md`.
- Conclusion: same formula, possibly with minor tweaks (L2/L3 thickness, tolerances or calculation basis). I used the **2023 drawing**: it is the current lens, and its curves are cleaner (fit residuals 0.01 pt vs 0.24 pt quantisation).
- **Open question:** the 2023 lens cut labels the **L5'/L5 doublet** (elements 5/6) "Focusing lens". The 2004 sheet and the patent name the rear doublet L6/L7 as the floating element. It is unclear whether the 2023 floating scheme changed or the label is loose. The lens file follows the patent.

## How the lens file was built

1. **Geometry.** I fitted circles to every surface of the 2023 vector lens cut, then took vertex positions, the stop plane, the image plane and drawn element heights.
   - Twelve surfaces are true spheres (residual ≤ 0.027 pt on a ~270 pt drawing).
   - Surface 7 (L4 front) is **not** a sphere: the sphere-fit residual is 0.26 pt. An even-asphere fit R + A4 brings it to 0.015 pt.
   - So Leica's drawing is a true CAD export. The drawn asphere is also the only way the patent's f(L4) = −369 mm can be met with real glass. The best-fit sphere would need n ≈ 1.38.
2. **Scale.** The drawing is scaled so that EFL = 51.6 mm, Leica's 2004 figure (the 2023 sheet gives none). The scale is 0.281465 mm/pt and depends weakly on the glasses.
3. **Glasses.** Leica publishes none. I fitted 8 × (nd, vd) by least squares with the geometry held fixed. The targets were:
   - polychromatic RMS spot (F, e, d, C) at 6 fields with real ray tracing;
   - Leica's distortion curve;
   - the patent partial focal lengths (±5 %) and best focus near the drawn image plane;
   - the patent glass claims, with a glass-map plausibility penalty.

   Three different starting sets converged to the same glasses. The asphere was refit at the same time but tied to the drawn profile (it moves < 2 µm).

   | Element | Fitted nd/vd | Nearest catalogue glass (illustrative only) | Patent / Leica constraint |
   |---|---|---|---|
   | L1 | 1.8247 / 42.0 | S-LAH55V 1.83481/42.74 | n > 1.80 ✓ |
   | L2 | 1.4733 / 87.3 | N-FK51A 1.48656/84.47, S-FPL52 1.456/90.3 | APD ✓ (FK type) |
   | L3 | 1.7364 / 35.2 | S-NBH51 1.7495/35.33, N-KZFS8 1.72047/34.70 | APD ✓ (KzFS type) |
   | L4 (asph) | 1.7524 / 51.8 | S-LAH97 / N-LAK33B 1.755/52.3 | — |
   | L5' | 1.7375 / 29.6 | N-SF1 1.71736/29.62, S-TIH13 1.74077/27.79 | — |
   | L5 | 1.9066 / 40.4 | none at this nd/vd; nearest S-LAH58 1.883/40.76 | n > 1.80 ✓ |
   | L6 | 1.8257 / 46.5 | S-LAH59 1.816/46.62 | n > 1.80 ✓ |
   | L7 | 1.5782 / 37.5 | S-FTM16 1.5927/35.31 | ν < 40 ✓ |

   Snapping to these catalogue glasses with only the asphere, focus and floating gap re-optimised gives 2–4× larger spots. I therefore kept the fitted values; the lens file only needs nd/vd. The dispersion model in the fit was a two-term Cauchy through nd and vd, so it has no anomalous partial dispersion and no secondary-spectrum fidelity.
4. **Focusing.** Patent claim 8: the whole lens moves forward by e and L6/L7 moves by r·e.
   - I measured r ≈ 0.77 from patent Fig. 2 vs 2a (front group shift 60 px, rear group 46 px).
   - In the reconstruction, r = 0.75–0.80 also minimises on-axis spherical aberration at 0.70 m (2.6–3 µm RMS, vs 18 µm at r = 1 and 30 µm at r = 0.5). This independently supports the reconstruction.
   - Results:
     - 0.70 m: e = 5.20 mm, magnification 1:11.29 (Leica: 1:11.3).
     - 0.45 m: e = 9.27 mm, magnification 1:6.37 (Leica 2023: 1:6.4).
   - The 0.45 m values extrapolate the 2004 law. See the "Focusing lens" open question above.

## Sag formula and units in the lens file

z(r) = c r² / (1 + √(1 − (1+k) c² r²)) + A4 r⁴ + A6 r⁶, with c = 1/R in mm and z positive towards the image.

- Asphere (row 7): R = −124.9816 mm, k = 0, A4 = −1.195732e−05, A6 = −5.329680e−09 (mm units). They come from the pt-unit fit by A_{2i}(mm) = A_{2i}(pt) · s^{1−2i}, with s = 0.281465. Patent and drawing have no other terms.
- The row carries `k= a4= a6=` inline, because the current `src/lens.rs` parser supports that. A `# asph …` comment line follows, as requested.
- Focus gaps are given as `# focus-gap` comments in two forms: distance from the image plane, which is Leica's convention, at 0.70 m and 0.45 m, for row 11 (floating gap) and row 14 (back focus).
- There is also an `@float 11 0.5106 2.6424 0.3644` line. The parser reads it as an object distance from the front vertex, and 0.3644 m from the vertex corresponds to 0.45 m from the image plane.
- The file was checked with `lens-render analyze`: EFL 51.601 mm, paraxial f/1.40, entrance pupil 25.19 mm behind S1.

## Validation against Leica data

| Quantity | Leica | Reconstruction |
|---|---|---|
| EFL | 51.6 mm (2004) | 51.60 (imposed) |
| Entrance pupil from S1 | 25.7 mm (2004) | 25.19 mm (not imposed in the final fit) |
| Entrance pupil before sensor | 25.4 mm before bayonet = 53.2 mm if flange 27.8 (2023) | 51.2 mm (2 mm off; reference plane uncertain) |
| Distortion at 19 / 21.6 mm | +0.32 / ~+0.54 % | +0.35 / +0.56 % (`analyze`) |
| Relative illumination, f/1.4, 5/10/15/20/21 mm | 86 / 67 / 46 / 26 / 23 % | 86 / 68 / 48 / 29 / 25 % (drawn apertures, cos⁴) |
| Relative illumination, f/2.8, 10/15/20 mm | 93 / 85 / 63 % | 95 / 89 / 68 % |
| Close-focus magnification | 1:11.3 @0.7 m; 1:6.4 @0.45 m | 1:11.29; 1:6.37 |
| Stop vs element sizes | — | The drawn stop is exactly paraxial f/1.40, and the drawn element edges clip at the same beam size, so the drawing is self-consistent. |

**MTF at f/1.4** (project's `lens-render mtf`, white light):

- On axis the reconstruction is much sharper than Leica: 40 lp/mm 0.92 vs 0.64, 20 lp/mm 0.97 vs 0.81.
- Leica's sagittal 40 lp/mm dip at 10–13 mm, recovering by 15–16 mm, appears in the reconstruction too, but deeper and earlier (≈0.01–0.15 between 8 and 15 mm).
- Corner values roughly match: at 21.6 mm, 5 lp/mm is S 0.65 / T 0.91 vs Leica S 0.57 / T 0.93 at 21 mm.
- At f/5.6 the reconstruction has 40 lp/mm centre 0.83 vs 0.77.

Interpretation: the glass fit traded field performance for on-axis correction. With only nd/vd and a Cauchy dispersion model, it also misses the real lens's secondary spectrum and spherochromatism. **Use Leica's MTF as a qualitative check, not a pass/fail target.** Matching it closely would need the real glasses, which are unknown. Fitting glasses to Leica's MTF is possible, but then that MTF is no longer independent validation.

## MTF refit

The first glass fit (above) was sharper than Leica on axis but clearly worse at mid-field: at f/1.4, 10 mm, sagittal 20 lp/mm 0.56 vs Leica's 0.73, and 40 lp/mm 0.09 vs 0.33. So the lens was refit to be at least as good as Leica's published MTF, which makes that MTF a fitting target, no longer an independent check.

- **Variables.** Glasses chosen from Schott/Ohara catalogue types near the old fits (APD glass N-FK51A in L2, as the patent says; n > 1.80 in L1, L5, L6), two continuous nd/vd glasses where no catalogue type fit; asphere A4–A10 within 2 µm of the drawn profile; radii within 1 % (R2 and R7 within 5 %); spacings within drawing precision; EFL, distortion, vignetting held.
- **Merit.** One-sided against Leica's 2023 MTF (S and T, 5/10/20/40 lp/mm, all heights, f/1.4, f/2.8, f/5.6): only shortfalls count (`mtf_target` in `docs/optimiser.md`).
- **Result.** 240 of 336 published points met. At f/1.4, 10 mm: sagittal 20 lp/mm 0.76 (Leica 0.73), 40 lp/mm 0.33 (0.33); tangential 0.85 / 0.59 (0.77 / 0.45). Centre 40 lp/mm 0.86 (Leica 0.64). Corner (21 mm) at or above Leica. The largest remaining gap: f/2.8, 10 mm, sagittal 40 lp/mm, 0.46 vs 0.65.
- **Cost.** White-light RMS at 2.5 m, f/1.4: centre 4.4 µm (was 2.5), 0.7 field 11.3 µm (13.6), corner 41 µm (32): a wider faint halo, while the corner MTF stays at or above Leica's. The refit was stopped at the best design found so far, not at a converged optimum.

## Uncertainties

- **Glasses.** Least certain; fitted, not known. The radii, spacings and asphere come from Leica's CAD drawing and are good to well under 1 %, assuming the brochure drawing is the production design.
- **Focal length.** 51.6 mm is from the 2004 sheet; the 2023 sheet does not state one. All lengths scale with it.
- **Asphere.** A6 is poorly determined by the drawing (A4-only and A4+A6 fits are equally good). A8+ are unknown and assumed 0.
- **Semi-diameters.** These are the drawn mechanical edges. Clear apertures are probably 0.2–0.5 mm smaller; the vignetting match suggests the drawn values are close to effective.
- **Floating law.** The 0.77 ratio comes from a hand-drawn patent figure, though the aberration optimum supports it. The 0.45 m values extrapolate. The 2023 "Focusing lens" label is unexplained.
- **Distortion offset.** Leica's 2023 distortion curve starts at +0.018 % at y' = 0, a plotting offset.

## Not found

- Any numerical prescription, from patent or third party.
- Glass types.
- Aspheric coefficients beyond what the drawing gives.
- A 2023-specific patent.
- Leica's exact focal length for the 2023 version.

---

## Sources

- Technical data sheet (EN), "Version date February 2023": https://leica-camera.com/sites/default/files/pm-14169-EN_Datenblatt-Summilux-M-50---Classic.pdf
- MTF images on the product page (same curves, raster 3840x2560): https://leica-camera.com/sites/default/files/2025-05/11728_Leica_Summilux-M-50_mtf01.jpg (f/1.4), `_mtf02.jpg` (f/2.8), `_mtf03.jpg` (f/5.6)
- 2004 (previous version, 11891/11892) brochure/data sheet, for comparison: https://www.summilux.net/m_system/images/Summilux50Asph.pdf

## Data sheet facts (2023)

| Item | Value |
|---|---|
| Angle of view (diag/horiz/vert), 24x36 | 45.4° / 38.4° / 26.1° |
| Lenses / groups | 8 / 5 |
| Aspherical surfaces | 1 |
| Entrance pupil position | 25.4 mm in front of the bayonet |
| Focus range | Live View 0.45 m – ∞; rangefinder 0.7 m – ∞ |
| Smallest object field / largest scale | 153 x 229 mm / 1:6.4 (at 0.45 m) |
| Diaphragm | f/1.4 – f/16, half stops, 11 blades |
| Filter | E46 |
| Length / diameter / weight | ~59.3 mm / ~58.6 mm / ~377 g |
| Focal length | not stated in the 2023 sheet; the 2004 sheet states **51.6 mm** |
| Lens cut label | "Focusing lens" arrow points at the cemented L5'/L5 doublet (elements 5/6) — see NOTES.md |

2004 sheet (previous version) for comparison: focal length 51.6 mm; 8/5; entrance pupil 25.7 mm "related to the first lens surface in light direction"; focus 0.7 m – ∞; smallest object field 271 x 407 mm; 1:11.3; angle of view 47°/40°/27°; floating element = rear cemented group (elements 7/8).

## MTF (2023 sheet)

Contrast in % vs image height y', infinity, white light; S = sagittal (solid), T = tangential (dashed).
Leica's text says max aperture, "2.0" (sic) and 5.6, but the charts are labelled f/1.4, f/2.8, f/5.6.

**Extraction method:** the PDF charts are vector graphics; each curve is a 70-point polyline. Values below are
linear interpolations of those polylines, calibrated on the chart frame (x: 0–21 mm, y: 0–100 %). Error is
essentially the PDF plotting resolution (≲0.3 %), i.e. far better than eyeball digitising. Frequency assignment
is by ranking curves within each S/T family (MTF decreases with frequency everywhere here; cross-checked against
the raster JPGs). Note the chart x-axis ends at **21 mm**, not 21.6 mm. Full data (27 heights, 0–21 mm) in
`summilux-50-asph-leica-mtf.json`.

### f/1.4, infinity, white light

| y' (mm) | 5 S | 5 T | 10 S | 10 T | 20 S | 20 T | 40 S | 40 T |
|---|---|---|---|---|---|---|---|---|
| 0 | 96.9 | 96.9 | 91.0 | 91.0 | 80.5 | 80.5 | 64.3 | 64.3 |
| 2.5 | 97.2 | 97.4 | 91.8 | 92.2 | 82.0 | 81.2 | 66.8 | 66.3 |
| 5 | 97.6 | 97.8 | 92.4 | 92.9 | 82.1 | 80.2 | 63.8 | 61.2 |
| 7.5 | 97.6 | 97.9 | 91.9 | 93.1 | 78.3 | 78.8 | 48.1 | 52.2 |
| 10 | 97.1 | 97.8 | 90.5 | 92.6 | 73.4 | 76.8 | 33.4 | 45.1 |
| 12.5 | 96.7 | 97.5 | 90.1 | 92.1 | 74.5 | 75.3 | 32.9 | 41.7 |
| 15 | 95.0 | 97.6 | 86.7 | 92.6 | 71.2 | 77.8 | 52.1 | 48.8 |
| 17.5 | 87.2 | 97.4 | 70.0 | 92.5 | 57.0 | 77.6 | 44.4 | 49.5 |
| 20 | 67.3 | 94.5 | 43.2 | 83.5 | 27.5 | 60.4 | 14.9 | 32.7 |
| 21 | 57.1 | 92.6 | 31.7 | 77.6 | 12.6 | 49.7 | 3.1 | 23.3 |

### f/2.8, infinity, white light

| y' (mm) | 5 S | 5 T | 10 S | 10 T | 20 S | 20 T | 40 S | 40 T |
|---|---|---|---|---|---|---|---|---|
| 0 | 98.6 | 98.6 | 96.6 | 96.6 | 91.2 | 91.2 | 76.0 | 76.0 |
| 2.5 | 98.6 | 98.4 | 96.9 | 96.4 | 92.1 | 90.5 | 79.0 | 74.1 |
| 5 | 98.6 | 98.3 | 97.2 | 95.9 | 93.2 | 88.2 | 82.6 | 68.1 |
| 7.5 | 98.6 | 98.1 | 96.7 | 94.7 | 91.5 | 84.1 | 77.7 | 57.2 |
| 10 | 98.3 | 97.6 | 95.5 | 92.9 | 87.2 | 78.3 | 64.9 | 44.1 |
| 12.5 | 98.1 | 97.2 | 94.8 | 91.6 | 84.6 | 73.7 | 56.5 | 33.9 |
| 15 | 98.4 | 97.6 | 95.9 | 92.8 | 89.3 | 77.9 | 71.3 | 44.4 |
| 17.5 | 98.7 | 97.7 | 96.1 | 93.2 | 87.1 | 78.9 | 65.9 | 50.7 |
| 20 | 91.3 | 93.8 | 73.3 | 81.4 | 43.4 | 58.9 | 26.4 | 36.7 |
| 21 | 86.2 | 91.4 | 57.9 | 73.6 | 16.4 | 46.6 | 6.6 | 28.1 |

### f/5.6, infinity, white light

| y' (mm) | 5 S | 5 T | 10 S | 10 T | 20 S | 20 T | 40 S | 40 T |
|---|---|---|---|---|---|---|---|---|
| 0 | 97.8 | 97.8 | 95.3 | 95.3 | 89.7 | 89.8 | 76.9 | 77.1 |
| 2.5 | 97.8 | 97.8 | 95.5 | 95.2 | 90.2 | 89.4 | 78.6 | 76.0 |
| 5 | 97.9 | 97.6 | 95.7 | 94.9 | 91.2 | 88.3 | 81.6 | 72.9 |
| 7.5 | 97.9 | 97.4 | 95.9 | 94.3 | 91.6 | 86.5 | 82.8 | 67.7 |
| 10 | 97.9 | 97.2 | 95.7 | 93.6 | 90.8 | 83.8 | 80.5 | 61.1 |
| 12.5 | 97.8 | 97.1 | 95.3 | 92.9 | 90.0 | 81.6 | 77.6 | 55.3 |
| 15 | 97.9 | 96.9 | 95.5 | 92.6 | 90.3 | 81.0 | 78.8 | 56.1 |
| 17.5 | 98.1 | 97.6 | 96.0 | 94.3 | 91.8 | 85.5 | 82.2 | 63.2 |
| 20 | 97.0 | 94.5 | 92.0 | 84.3 | 78.8 | 61.6 | 52.4 | 39.0 |
| 21 | 96.2 | 92.2 | 89.0 | 76.6 | 69.5 | 43.3 | 31.2 | 20.0 |

## Vignetting (2023 sheet, vector-extracted)

Relative illumination in % (100 % = none), infinity. Curve styles: dotted = f/1.4, dash-dot = f/2.8, solid = f/5.6
(identified by end-point labels in the chart). Presumably includes cos⁴ falloff (Leica: "decrease of image brightness
(irradiance) towards the edges").

| y' (mm) | 0 | 2.5 | 5 | 7.5 | 10 | 12.5 | 15 | 17.5 | 20 | 21 |
|---|---|---|---|---|---|---|---|---|---|---|
| f/1.4 | 100 | 94.5 | 86.1 | 76.8 | 66.6 | 56.0 | 45.5 | 35.4 | 26.0 | 22.6 |
| f/2.8 | 100 | 99.5 | 98.3 | 96.0 | 93.0 | 89.3 | 85.0 | 76.7 | 62.9 | 57.2 |
| f/5.6 | 100 | 99.5 | 98.1 | 95.9 | 92.8 | 89.0 | 84.4 | 79.3 | 73.4 | 71.0 |

## Distortion (2023 sheet, vector-extracted)

Relative distortion (%), pincushion positive. The curve sits at +0.018 % at y'=0 (plotting offset); values below
are raw, subtract ~0.02 for true value.

| y' (mm) | 0 | 5 | 10 | 12 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dist. % | 0.02 | 0.02 | 0.05 | 0.09 | 0.12 | 0.15 | 0.18 | 0.22 | 0.28 | 0.34 | 0.41 | 0.50 |

## 2004 vs 2023 MTF

The 2004 brochure charts are also vector (extracted the same way; x-axis 0–~21 mm, 5 mm grid). Curve shapes are
the same (same dips/peaks at the same heights), but the 2023 curves are higher on-axis at 20/40 lp/mm, e.g. f/1.4
40 lp/mm centre 56.4 → 64.3 %, f/2.8 64 → 76 %, f/5.6 70 → 77 %. Off-axis the differences are a few %, both
signs. Table (old value, [2023 value] in brackets):

```
== OLD (2004)  f/1.4    [new 2023 value in brackets]
y_mm           5S           5T          10S          10T          20S          20T          40S          40T
 0.0  96.9 [ 96.9]  96.9 [ 96.9]  90.3 [ 91.0]  90.3 [ 91.0]  76.2 [ 80.5]  76.2 [ 80.5]  56.4 [ 64.3]  56.4 [ 64.3]
 2.5  97.2 [ 97.2]  97.4 [ 97.4]  91.4 [ 91.8]  91.8 [ 92.2]  78.2 [ 82.0]  77.8 [ 81.2]  59.8 [ 66.8]  57.5 [ 66.3]
 5.0  97.6 [ 97.6]  97.6 [ 97.8]  92.2 [ 92.4]  92.2 [ 92.9]  79.5 [ 82.1]  77.1 [ 80.2]  59.4 [ 63.8]  51.3 [ 61.2]
 7.5  97.6 [ 97.6]  97.8 [ 97.9]  91.9 [ 91.9]  92.2 [ 93.1]  76.9 [ 78.3]  75.9 [ 78.8]  47.8 [ 48.1]  44.3 [ 52.2]
10.0  97.2 [ 97.1]  97.6 [ 97.8]  90.5 [ 90.5]  91.9 [ 92.6]  72.8 [ 73.4]  74.7 [ 76.8]  35.0 [ 33.4]  40.4 [ 45.1]
12.5  97.1 [ 96.7]  97.4 [ 97.5]  90.9 [ 90.1]  92.0 [ 92.1]  75.7 [ 74.5]  75.3 [ 75.3]  37.8 [ 32.9]  42.1 [ 41.7]
15.0  95.6 [ 95.0]  97.6 [ 97.6]  88.0 [ 86.7]  92.8 [ 92.6]  73.5 [ 71.2]  78.7 [ 77.8]  49.8 [ 52.1]  51.5 [ 48.8]
17.5  88.2 [ 87.2]  97.2 [ 97.4]  71.7 [ 70.0]  91.5 [ 92.5]  59.1 [ 57.0]  76.4 [ 77.6]  47.5 [ 44.4]  49.8 [ 49.5]
20.0  68.8 [ 67.3]  93.9 [ 94.5]  43.5 [ 43.2]  81.4 [ 83.5]  24.2 [ 27.5]  54.0 [ 60.4]   9.5 [ 14.9]  26.8 [ 32.7]
== OLD (2004)  f/2.8    [new 2023 value in brackets]
y_mm           5S           5T          10S          10T          20S          20T          40S          40T
 0.0  98.3 [ 98.6]  98.3 [ 98.6]  95.5 [ 96.6]  95.5 [ 96.6]  87.1 [ 91.2]  87.1 [ 91.2]  63.8 [ 76.0]  64.0 [ 76.0]
 2.5  98.5 [ 98.6]  98.3 [ 98.4]  95.9 [ 96.9]  95.3 [ 96.4]  88.7 [ 92.1]  86.3 [ 90.5]  68.8 [ 79.0]  62.2 [ 74.1]
 5.0  98.6 [ 98.6]  98.1 [ 98.3]  96.8 [ 97.2]  94.7 [ 95.9]  92.0 [ 93.2]  84.3 [ 88.2]  78.3 [ 82.6]  57.3 [ 68.1]
 7.5  98.6 [ 98.6]  97.8 [ 98.1]  97.1 [ 96.7]  93.6 [ 94.7]  92.6 [ 91.5]  80.7 [ 84.1]  80.7 [ 77.7]  49.6 [ 57.2]
10.0  98.5 [ 98.3]  97.4 [ 97.6]  96.1 [ 95.5]  92.3 [ 92.9]  89.5 [ 87.2]  76.6 [ 78.3]  71.3 [ 64.9]  41.2 [ 44.1]
12.5  98.3 [ 98.1]  97.4 [ 97.2]  95.3 [ 94.8]  92.0 [ 91.6]  86.6 [ 84.6]  75.0 [ 73.7]  63.0 [ 56.5]  37.5 [ 33.9]
15.0  98.6 [ 98.4]  97.8 [ 97.6]  96.2 [ 95.9]  93.6 [ 92.8]  89.7 [ 89.3]  80.6 [ 77.9]  70.9 [ 71.3]  49.2 [ 44.4]
17.5  98.5 [ 98.7]  97.5 [ 97.7]  95.8 [ 96.1]  92.8 [ 93.2]  88.5 [ 87.1]  79.6 [ 78.9]  74.5 [ 65.9]  53.7 [ 50.7]
20.0  92.9 [ 91.3]  93.5 [ 93.8]  77.1 [ 73.3]  80.0 [ 81.4]  42.6 [ 43.4]  53.5 [ 58.9]  11.6 [ 26.4]  35.5 [ 36.7]
== OLD (2004)  f/5.6    [new 2023 value in brackets]
y_mm           5S           5T          10S          10T          20S          20T          40S          40T
 0.0  97.6 [ 97.8]  97.6 [ 97.8]  94.7 [ 95.3]  94.7 [ 95.3]  87.4 [ 89.7]  87.6 [ 89.8]  70.3 [ 76.9]  70.5 [ 77.1]
 2.5  97.6 [ 97.8]  97.6 [ 97.8]  94.9 [ 95.5]  94.6 [ 95.2]  88.3 [ 90.2]  87.2 [ 89.4]  72.8 [ 78.6]  69.5 [ 76.0]
 5.0  97.8 [ 97.9]  97.4 [ 97.6]  95.5 [ 95.7]  94.3 [ 94.9]  90.1 [ 91.2]  86.1 [ 88.3]  78.0 [ 81.6]  66.8 [ 72.9]
 7.5  97.9 [ 97.9]  97.2 [ 97.4]  95.9 [ 95.9]  93.8 [ 94.3]  91.4 [ 91.6]  84.6 [ 86.5]  82.1 [ 82.8]  62.7 [ 67.7]
10.0  97.9 [ 97.9]  97.2 [ 97.2]  95.9 [ 95.7]  93.2 [ 93.6]  91.4 [ 90.8]  82.7 [ 83.8]  82.1 [ 80.5]  58.2 [ 61.1]
12.5  97.9 [ 97.8]  97.1 [ 97.1]  95.6 [ 95.3]  92.9 [ 92.9]  90.7 [ 90.0]  81.7 [ 81.6]  79.7 [ 77.6]  55.7 [ 55.3]
15.0  97.9 [ 97.9]  97.2 [ 96.9]  95.7 [ 95.5]  93.3 [ 92.6]  90.9 [ 90.3]  83.1 [ 81.0]  80.2 [ 78.8]  59.4 [ 56.1]
17.5  98.1 [ 98.1]  97.4 [ 97.6]  95.9 [ 96.0]  94.1 [ 94.3]  91.1 [ 91.8]  86.0 [ 85.5]  81.2 [ 82.2]  66.6 [ 63.2]
20.0  97.3 [ 97.0]  95.5 [ 94.5]  93.1 [ 92.0]  87.2 [ 84.3]  81.7 [ 78.8]  66.0 [ 61.6]  56.4 [ 52.4]  33.0 [ 39.0]
```
