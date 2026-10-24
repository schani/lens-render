//! Viewer panel for the lens itself: cross-section, analyses and the
//! prescription.

mod diagram;
mod glassmap;
mod mtfview;
mod plots;
mod psfview;
mod table;

pub use diagram::{Diagram, FocusTravel, Overlays, RayPath, focus_travel, trace_fans};
pub use glassmap::GlassMap;
pub use mtfview::{mtf_ui, published};
pub use plots::{footprint_plots, light_loss_chart, seidel_chart};
pub use psfview::PsfView;
pub use table::LensEditor;
