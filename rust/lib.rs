//! Python extension module for bounded destination-plan search and validation.

mod api;
mod errors;
mod input;
mod model;
mod oracle;
mod output;
mod scoring;
mod top_k;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    api::register(module)
}
