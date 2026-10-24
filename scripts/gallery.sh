#!/bin/sh
# Renders the demo scenes into docs/ as JPEGs.
set -e
cd "$(dirname "$0")/.."
cargo build --release
bin=./target/release/lens-render
out=docs
tmp=$(mktemp -d)
mkdir -p "$out"

r() { name=$1; shift; "$bin" render "$@" -o "$tmp/$name.png"; ffmpeg -loglevel error -y -i "$tmp/$name.png" -q:v 2 "$out/$name.jpg"; }

r chart scenes/chart.toml --set width=1800 --set spp=256
r night scenes/night.toml --set width=1800 --set spp=256
r bokeh-f3.5 scenes/bokeh.toml --set width=1200
r bokeh-f8 scenes/bokeh.toml --set width=1200 --set fstop=8 --set exposure_ev=-2.4
r flare scenes/flare.toml --set width=1200
r garden-coated scenes/garden.toml --set width=1500 --set spp=256
r garden-uncoated scenes/garden.toml --set width=1500 --set spp=256 --set coating=none
"$bin" analyze lenses/cooke-triplet.lens --out "$tmp/triplet" | grep -v "^Wrote" > "$out/triplet-analysis.txt"
cp "$tmp/triplet-layout.png" "$tmp/triplet-spots.png" "$out/"
rm -rf "$tmp"
