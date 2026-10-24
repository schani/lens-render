//! Viewer panel for the lens itself: cross-section, analyses and the
//! prescription.

mod diagram;
mod glassmap;
mod psfview;
mod table;

pub use diagram::{Diagram, FocusTravel, Overlays, RayPath, focus_travel, trace_fans};
pub use glassmap::GlassMap;
pub use psfview::PsfView;
pub use table::LensEditor;
