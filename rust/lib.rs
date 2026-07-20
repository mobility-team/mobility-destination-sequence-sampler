//! Python extension module for bounded destination-plan search and validation.

mod api;
mod bidirectional;
mod errors;
mod factor_tree;
mod input;
mod kernel_experiment;
mod model;
mod output;
mod particle;
mod profile;
mod sampler;
mod second_order;
mod ternary_reference;

use pyo3::prelude::*;

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    api::register(module)
}
