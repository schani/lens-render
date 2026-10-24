//! Edits of a lens prescription, as the graphical editor makes them. Rows
//! are surfaces; a group is a run of surfaces enclosing glass (an element,
//! or a cemented group), given by its first and last surface.

use crate::glass::Glass;
use crate::lens::{Lens, Surface};

/// The group containing surface `i`, if `i` borders glass.
pub fn group_of(lens: &Lens, i: usize) -> Option<(usize, usize)> {
    let glass = |k: usize| lens.surfaces[k].material.is_some();
    let inside = glass(i) || (i > 0 && glass(i - 1));
    if !inside {
        return None;
    }
    let mut first = i;
    while first > 0 && glass(first - 1) {
        first -= 1;
    }
    let mut last = i;
    while glass(last) && last + 1 < lens.surfaces.len() {
        last += 1;
    }
    Some((first, last))
}

/// Changes the gap after row `row` by `dz`, floating gaps included.
fn add_gap(lens: &mut Lens, row: usize, dz: f64) {
    lens.surfaces[row].thickness += dz;
    for f in lens.floating.iter_mut().filter(|f| f.surface == row) {
        f.at_infinity += dz;
        f.at_close += dz;
    }
}

/// Room to move surfaces `first..=last` by `dz`: gaps stay non-negative.
fn clamp_move(lens: &Lens, first: usize, last: usize, dz: f64) -> f64 {
    let gap = |row: usize| {
        let s = lens.surfaces[row].thickness;
        lens.floating.iter().filter(|f| f.surface == row).fold(s, |m, f| m.min(f.at_infinity).min(f.at_close))
    };
    let back = if first > 0 { -gap(first - 1) } else { f64::NEG_INFINITY };
    let ahead = if last + 1 < lens.surfaces.len() { gap(last) } else { f64::INFINITY };
    dz.clamp(back, ahead)
}

/// Moves surfaces `first..=last` along the axis by `dz` (mm), taking the
/// room from the gaps on either side.
pub fn shift(lens: &mut Lens, first: usize, last: usize, dz: f64) {
    let dz = clamp_move(lens, first, last, dz);
    if first > 0 {
        add_gap(lens, first - 1, dz);
    }
    add_gap(lens, last, -dz);
}

/// Bends surface `i` so that its rim, at height `sd`, sags by `sag` (mm).
pub fn set_rim(lens: &mut Lens, i: usize, sag: f64, sd: f64) {
    let s = &mut lens.surfaces[i];
    s.semi_diameter = sd.max(0.2);
    if s.is_stop {
        return;
    }
    let c = 2.0 * sag / (sag * sag + s.semi_diameter * s.semi_diameter);
    s.radius = if c.abs() < 1e-5 { 0.0 } else { 1.0 / c };
}

/// Inserts a thin N-BK7 biconvex element after surface `i` (in its air gap).
pub fn insert_element(lens: &mut Lens, i: usize) {
    let sd = lens.surfaces[i].semi_diameter;
    let gap = lens.surfaces[i].thickness;
    let t = (0.3 * gap).clamp(1.0, 4.0);
    let surface = |radius: f64, thickness: f64, material: Option<Glass>| Surface {
        radius,
        thickness,
        material,
        semi_diameter: sd,
        is_stop: false,
        z: 0.0,
        decenter: (0.0, 0.0),
        conic: 0.0,
        aspheric: Vec::new(),
    };
    let after = ((gap - t) / 2.0).max(0.5);
    lens.surfaces[i].thickness = after;
    lens.insert_surface(i + 1, surface(6.0 * sd, t, Glass::parse("N-BK7").ok()));
    lens.insert_surface(i + 2, surface(-6.0 * sd, after, None));
}

/// Removes the group `first..=last`, leaving the following surfaces in place.
pub fn delete_group(lens: &mut Lens, first: usize, last: usize) {
    if (first..=last).any(|k| lens.surfaces[k].is_stop) || lens.surfaces.len() - (last - first + 1) < 2 {
        return;
    }
    let length: f64 = (first..=last).map(|k| lens.surfaces[k].thickness).sum();
    if first > 0 {
        add_gap(lens, first - 1, length);
    }
    for k in (first..=last).rev() {
        lens.remove_surface(k);
    }
}

/// Turns the group `first..=last` around.
pub fn flip_group(lens: &mut Lens, first: usize, last: usize) {
    let old: Vec<Surface> = lens.surfaces[first..=last].to_vec();
    let n = old.len();
    for k in 0..n {
        let src = &old[n - 1 - k];
        let s = &mut lens.surfaces[first + k];
        s.radius = -src.radius;
        s.conic = src.conic;
        s.aspheric = src.aspheric.iter().map(|a| -a).collect();
        s.semi_diameter = src.semi_diameter;
        if k + 1 < n {
            // The medium and thickness after surface k were after surface n − 2 − k.
            s.material = old[n - 2 - k].material.clone();
            s.thickness = old[n - 2 - k].thickness;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triplet() -> Lens {
        Lens::parse(include_str!("../lenses/cooke-triplet.lens")).unwrap()
    }

    #[test]
    fn groups_and_moves() {
        let mut l = triplet();
        assert_eq!(group_of(&l, 0), Some((0, 1)));
        assert_eq!(group_of(&l, 1), Some((0, 1)));
        assert_eq!(group_of(&l, 2), Some((2, 3)));
        let before: Vec<f64> = l.surfaces.iter().map(|s| s.thickness).collect();
        shift(&mut l, 2, 3, 0.5);
        assert!((l.surfaces[1].thickness - before[1] - 0.5).abs() < 1e-12);
        assert!((l.surfaces[3].thickness - before[3] + 0.5).abs() < 1e-12);
        // Can't move into the next element.
        shift(&mut l, 2, 3, 100.0);
        assert_eq!(l.surfaces[3].thickness, 0.0);
    }

    #[test]
    fn flip_twice_is_identity() {
        let mut l = triplet();
        let text = l.to_text();
        flip_group(&mut l, 0, 1);
        assert_ne!(l.to_text(), text);
        flip_group(&mut l, 0, 1);
        assert_eq!(l.to_text(), text);
    }

    #[test]
    fn insert_and_delete_keep_the_rest_in_place() {
        let mut l = triplet();
        let z_last = l.surfaces.last().unwrap().z;
        insert_element(&mut l, 1);
        let mut l = Lens::parse(&l.to_text()).unwrap();
        assert_eq!(l.surfaces.len(), 9);
        delete_group(&mut l, 2, 3);
        let l = Lens::parse(&l.to_text()).unwrap();
        assert!((l.surfaces.last().unwrap().z - z_last).abs() < 1e-9);
    }

    #[test]
    fn rim_sag_sets_curvature() {
        let mut l = triplet();
        set_rim(&mut l, 0, 1.0, 10.0);
        assert!((l.surfaces[0].sag(10.0) - 1.0).abs() < 1e-9);
    }
}
