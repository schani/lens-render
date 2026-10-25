#!/bin/sh
# Regenerates the explainer lenses (lenses/explainer/*.lens) from their job
# files, the singlet bending curve and the numbers table.
set -e
cd "$(dirname "$0")/.."
cargo build --release
bin=./target/release/lens-render
d=lenses/explainer
for job in "$d"/*.toml; do
    "$bin" optimize "$job"
done
"$bin" bending "$d/01-singlet.lens" --fstop 4 --focus 2.5 --out "$d/bending.json"
"$bin" numbers --focus 2.5 --json "$d/numbers.json" \
    "$d/01-singlet.lens@4" "$d/02-best-form.lens@4" "$d/02b-flat-front.lens@4" "$d/02-best-form.lens@16" \
    "$d/04-achromat.lens@8" "$d/05-landscape.lens@14" "$d/06-rapid-rectilinear.lens@8" \
    lenses/cooke-triplet.lens@3.5 "$d/08-triplet-f2.lens@2" "$d/09-clark.lens@4" "$d/10-planar.lens@2.8" \
    lenses/double-gauss.lens@max "$d/11b-double-gauss.lens@2"
