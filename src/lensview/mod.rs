//! Viewer panel for the lens itself: cross-section, analyses and the
//! prescription.

mod diagram;
mod table;

pub use diagram::{RayPath, diagram, trace_fans};
pub use table::LensEditor;
