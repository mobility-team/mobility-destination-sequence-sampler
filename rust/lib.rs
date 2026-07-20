//! Python extension module for bounded destination-plan search and validation.

mod api;
mod bidirectional;
mod common;
mod errors;
mod input;
mod model;
mod output;
mod particle;
mod ternary_reference;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    api::register(module)
}
