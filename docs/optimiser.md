# Optimiser job files

`lens-render optimize <job.toml>` optimises a start design by damped least
squares (Levenberg–Marquardt, finite-difference Jacobian) and writes the
result next to the job file. `src/optimize.rs` has the details;
`lenses/explainer/*.toml` are examples of the basic options.

## Basics

```toml
name = "My lens"                  # @name of the output
header = """…"""                  # comment at the top of the output
output = "my-lens.lens"           # relative to the job file
fstop = 2                         # iris set to this f-number
focus = 2.5                       # object distance, m
efl = 50                          # held exactly by scaling all curvatures
fields = [0.0, 0.5, 0.7, 1.0]     # × 21.63 mm (half-diagonal of 36 × 24)
field_weights = [1, 1, 1, 1]
curvatures = [1, 2, 4, 5]         # rows (1-based) that vary
thicknesses = [1, 2, 3, 4]
symmetric = false                 # mirror the rows in front of the stop
iterations = 200
restarts = 0                      # random restarts around the best design
restart_scale = 0.15
start = """
  30.0   6.0   S-LAH79    15
  60.0   5.0   air        13
  stop   6.0   air         9
 -60.0   5.0   N-PK52A    12
 -25.0  35     air        13
"""
```

Clear apertures always vary (soft-edged while optimising).

## Merit

The merit is the sum of squared residuals: the white-light RMS spot (µm)
per field from real rays on the sensor plane the renderer focuses to, plus
penalties (1000 per mm of violation unless noted).

| Option | Default | Meaning |
| --- | --- | --- |
| `wavelengths = [[nm, w], …]` | F, e, C 1:2:1 | merit wavelengths and weights (`mono = nm` for one) |
| `rms_weight` | 1 | weight of the RMS spot (0: other terms only) |
| `mtf_weight` | 0 (off) | geometric MTF: µm of RMS per unit of lost contrast |
| `mtf_freqs` | [10, 30] | lp/mm, sagittal and tangential, per merit field |
| `vignetting` | 0 | corner may lose this fraction of the light, ∝ h² |
| `max_distortion` | – | \|distortion\| in % at every field |
| `min_corner_illumination` | – | relative illumination at 1.0 (cos⁴, vignetting, distortion) |
| `max_length` | – | front vertex to sensor, mm |
| `min_back_focus` | 15 | last vertex to sensor, mm |
| `min_glass_centre`, `min_glass_edge`, `max_glass`, `min_air` | 1.5, 1, 20, 0.1 | mm |
| `thickness_limits = [[row, min, max]]` | – | single rows |
| `[[material_limits]]` `glass`, `max_thickness`, `max_departure`, `max_semi_diameter` | – | elements of one material (e.g. a crystal hard to figure), wherever the glass choice puts it |
| `seidel_weights` | – | weights on W040 … W111 at full field |

The geometric MTF is the Fourier transform of the weighted spot (no
diffraction): cheap and smooth, fine for steering a design. Distortion
compares the chief ray with the paraxial image height. Field points sit at
paraxially estimated angles and apertures are soft, so distortion and
illumination come out within ~0.05 % and ~0.03 of `lens-render numbers`.

## MTF targets

```toml
[mtf_target]
reference = "summilux-50-asph-leica-mtf.json"   # as for `lens-render mtf --reference`
fstops = [1.4, 2.8, 5.6]          # keys "f/1.4" … in the reference
heights = [0, 5, 10, 15, 20, 21]  # mm, interpolated; default: the reference's
weight = 10                       # µm of RMS per unit of contrast short
margin = 0.01                     # contrast to exceed the reference by
grid = 32                         # pupil rays per side
```

One-sided: at every f-number, height, frequency and azimuth (the
reference's keys, e.g. `40S`, `40T`) where the design falls short of
reference + margin, the shortfall × `weight` is a residual; where it is
better, nothing. Each f-number gets its iris and the renderer's focus.
The object sits at the job's `focus` (use e.g. 1000 m for infinity). The
contrast is the geometric MTF of the white-light spot (CIE ȳ, 430–670 nm)
times the diffraction limit at the working f-number; on the Summilux it
stays within ~0.05 of the wave-optical MTF of `lens-render mtf` and tends
to read low where the S or T focus lies far off the sensor (up to 0.09).
Logs and the output header report how many targets are met and the
largest shortfall.

## Fitting a known geometry

For reconstructions (a drawn design, unknown glasses):

| Option | Default | Meaning |
| --- | --- | --- |
| `vary_apertures` | true | false: clear apertures stay as drawn (and `vignetting` is ignored) |
| `max_radius_change` | – | relative, from the start design at the job's focal length |
| `radius_change_limits = [[row, change]]` | – | single rows |
| `max_profile_change` | – | mm of sag from the start profile over the clear aperture, aspheric rows; 1 µm ≡ 10 µm of RMS |

Thicknesses are held with `thickness_limits`.

## Aspheres

```toml
conics = [1, 4]                   # rows whose conic constant varies
aspheres = [2, 5]                 # rows whose a4 … vary
asphere_orders = [4, 6, 8, 10]    # default; up to 18
max_departure = 0.5               # mm from the sphere through vertex and rim
max_departure_slope = 0.1         # mm/mm, slope of that departure
max_inflections = 1               # sign changes of the profile's curvature
max_surface_slope = 60            # degrees (default), aspheres only
```

Each aspheric coefficient is a variable scaled to its sag at the start
design's rim, aₙ hⁿ (mm), the conic to its approximate rim sag c³h⁴k/8, so
the finite-difference steps (1 µm of sag) and restarts are comparable
between orders. On aspheric surfaces also: glass and air thickness across
the whole aperture (16 radii, not just centre and rim), the clear aperture
inside the conic's domain, the surface slope limit. A departure counts 10×
(1 µm ≡ 10 µm of RMS). Too many inflections are penalised by the size of the
weakest curvature lobes. Outputs carry `k=` and `aN=` columns.

## Spectral stop (experimental)

The start design's stop row may carry `annulus=<fraction> pass=<lo>-<hi>`:
inside fraction × the stop's semi-diameter all light passes, outside only
`lo`–`hi` nm (edges 10 nm wide). Merit rays are weighted by its
transmission (the annulus edge soft while optimising).

```toml
spectral_stop = ["inner", "lo", "hi"]   # any of them vary
min_stop_transmission = 0.6             # luminous (CIE ȳ), wide open
wavelengths = [[440, 1], [450, 1], …]   # band edges need wavelengths within 5 nm
```

## Glass choice and global search

```toml
[[glass_choice]]
row = 1
glasses = ["S-LAH79", "N-LASF31A", "N-LAK9"]

[[glass_choice]]
row = 4
glasses = ["N-PK52A", "CAF2"]

[global]
restarts = 64          # local optimisations, in parallel over all cores
keep = 5               # best distinct designs written
seed = 1
perturb = 0.3          # relative, curvatures and thicknesses
asphere_perturb = 0.02 # mm of rim sag
hops = 0               # basin hops per start (perturb / 2, keep if better)
iterations = 200       # per descent (default: the job's)
distinct = 0.01        # see below
```

Start r descends from a random perturbation of the start design (start 0
as given) with glass combination r mod (number of combinations), the first
`glass_choice` row cycling fastest, so with enough restarts every
combination is tried. Curvatures get ± `perturb` relative and ± `perturb` /
efl absolute (flat surfaces bend). Each start draws from its own random
stream of `seed`, so results do not depend on thread scheduling. Every
local minimum is a candidate; two count as one if they use the same
glasses and differ by at most `distinct` / efl in every curvature, 10 ×
`distinct` mm in thickness, `distinct` mm of rim sag in aspheric terms.

The best design goes to `output`, the next ones to `<stem>-2.lens` …
`<stem>-<keep>.lens`, each with its merit, rank and start in the header.
Without `[global]`, `glass_choice` is ignored.

Timing (Apple M2 Max, 12 cores): a two-element f/2 design with four
aspheric surfaces (22 variables, MTF, distortion and illumination terms)
descends in about 5 s; 64 restarts take about 150 s.
