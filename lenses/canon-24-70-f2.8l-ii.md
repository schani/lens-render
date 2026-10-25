# Canon zoom 24–70 mm f/2.8 — US 8,736,971 example 3

## Source

| What | URL |
|---|---|
| Patent | US 8,736,971 B2 (Canon, priority JP 2012-023338), numerical example 3: https://patents.google.com/patent/US8736971B2/en |
| Granted PDF (example 3: cols. 12–13) | https://patentimages.storage.googleapis.com/b1/74/69/cdda2886ea7380/US8736971.pdf |
| Lens identification | Bill Claff, Optical Bench Hub: `US008736971 / Example03P` = "Canon EF24-70mm f2.8L II USM", https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/OpticalBenchHub.htm |
| Production lens | Canon Camera Museum, EF24-70mm f/2.8L II USM: https://global.canon/en/c-museum/product/ef424.html |

The example has the production lens' construction: 18 elements in 13 groups,
one Super UD element (nd 1.43875, νd 94.9), two UD elements (1.49700/81.5) and
three aspherical elements (surfaces 6, 22, 29/30). Examples 1 and 2 of the
patent use a resin aspheric layer instead.

## Taken from the patent

- Radii, thicknesses, nd/νd of all 33 surfaces.
- Aspheric coefficients K, A4–A12. The patent's sag formula is ours:
  x = (h²/R) / (1 + √(1 − (1+k)(h/R)²)) + A4h⁴ + … + A12h¹², k the conic
  constant (all 0 here). The printed formula writes every term as h² (a
  typo); the coefficient names give the orders.
- Zoom positions f = 24.70 / 34.93 / 67.88 mm, F2.91 throughout, the variable
  gaps d5, d15, d24 and d33 (back focus).

## Assumed or changed

- Surface 16 (flat, 1.90 mm in front of the stop, presumably a flare cutter)
  is dropped and its 1.90 mm added to d15. This keeps the lens at 32 rows.
- The stop sits 0.01 mm in front of surface 18 (patent 0.00). Chief rays
  crossing the stop centre would otherwise start on the next vertex, which
  the tracer rejects.
- Semi-diameters are not given. They come from tracing the f/2.91 axial
  bundle and the full corner bundles (fields 0.5–1.0 × 21.64 mm, the whole
  stop) at 13 focal lengths across the range. Each is the largest ray height
  plus 0.3 mm, cut where neighbouring surfaces meet (rows 2–4, 7–8, 12–13,
  21–22, 29–30). So there is no mechanical vignetting at f/2.91 beyond what
  the patent's own geometry forces. Front element: 38.8 mm (77.6 mm clear
  diameter; the lens takes 82 mm filters).
- Stop: `@zoom-fnumber 2.91 2.91 2.91`. The opening follows the f-number,
  r = 9.52 / 10.59 / 12.94 mm at the three positions. A fixed stop would make
  the lens f/2.1 at 24 mm.
- Between the patent's positions, gaps lie on the parabola through the three
  positions in log focal length. Canon's cams are not published.
- Iris: 9 blades (Canon spec), curvature 0.8 (guess for Canon's "circular
  aperture"). Multi-coated.
- Focusing moves the whole lens (the simulator's unit focusing). The real lens
  focuses internally (to 0.38 m), so close-range aberrations and breathing
  differ.
- Glass dispersion is a two-term Cauchy through nd/νd. Anomalous partial
  dispersion of the UD and Super UD glasses, and with it the secondary
  spectrum, is not modelled.

## Verification (`cargo test zoom_matches_patent`, `analyze --focal`)

| | Wide | Middle | Tele |
|---|---|---|---|
| Patent f (mm) | 24.70 | 34.93 | 67.88 |
| Paraxial EFL (mm) | 24.697 | 34.940 | 67.917 |
| Patent BF (mm) | 38.08 | 46.62 | 63.32 |
| Paraxial BFD / best focus (mm) | 38.084 / 38.044 | 46.626 / 46.601 | 63.347 / 63.345 |
| F-number | 2.91 | 2.91 | 2.91 |
| On-axis RMS spot, f/2.91 (µm) | 1.9 | 1.5 | 2.1 |
| MTF 10 / 30 lp/mm, centre, f/2.91 | 0.97 / 0.91 | 0.98 / 0.92 | 0.97 / 0.91 |
| MTF 10 / 30 lp/mm at 15 mm, S / T | 0.96, 0.81 / 0.89, 0.60 | 0.95, 0.70 / 0.90, 0.43 | 0.94, 0.72 / 0.92, 0.67 |
| Distortion near the corner (21–22 mm) | −3.2 % | +0.9 % | +2.5 % |
| Relative illumination there, f/2.91 | 44 % | 61 % | 56 % |

The total length (front vertex to best focus) is 154.51 / 162.02 / 185.78 mm;
the patent gives 154.56 / 162.05 / 185.77.

The patent's aberration data are only drawn as curves (FIG. 6), so there are
no numbers to compare.
