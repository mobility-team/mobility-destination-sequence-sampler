use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Clone, Debug, Default)]
pub struct OutputTable {
    pub context_id: Vec<u64>,
    pub draw_id: Vec<u32>,
    pub layer: Vec<u32>,
    pub origin: Vec<u32>,
    pub destination: Vec<u32>,
    pub local_log_weight: Vec<f64>,
    pub total_log_weight: Vec<f64>,
    /// Populated only by the particle proposal path. Exact samplers retain
    /// NaN here rather than changing their established result contract.
    pub proposal_log_probability: Vec<f64>,
    pub importance_log_weight: Vec<f64>,
}

pub struct OutputRow {
    pub context_id: u64,
    pub draw_id: u32,
    pub layer: u32,
    pub origin: u32,
    pub destination: u32,
    pub local_log_weight: f64,
    pub total_log_weight: f64,
}

impl OutputTable {
    pub fn push(&mut self, row: OutputRow) {
        self.context_id.push(row.context_id);
        self.draw_id.push(row.draw_id);
        self.layer.push(row.layer);
        self.origin.push(row.origin);
        self.destination.push(row.destination);
        self.local_log_weight.push(row.local_log_weight);
        self.total_log_weight.push(row.total_log_weight);
        self.proposal_log_probability.push(f64::NAN);
        self.importance_log_weight.push(f64::NAN);
    }

    pub fn push_particle(
        &mut self,
        row: OutputRow,
        proposal_log_probability: f64,
        importance_log_weight: f64,
    ) {
        self.context_id.push(row.context_id);
        self.draw_id.push(row.draw_id);
        self.layer.push(row.layer);
        self.origin.push(row.origin);
        self.destination.push(row.destination);
        self.local_log_weight.push(row.local_log_weight);
        self.total_log_weight.push(row.total_log_weight);
        self.proposal_log_probability.push(proposal_log_probability);
        self.importance_log_weight.push(importance_log_weight);
    }

    pub fn extend(&mut self, other: Self) {
        self.context_id.extend(other.context_id);
        self.draw_id.extend(other.draw_id);
        self.layer.extend(other.layer);
        self.origin.extend(other.origin);
        self.destination.extend(other.destination);
        self.local_log_weight.extend(other.local_log_weight);
        self.total_log_weight.extend(other.total_log_weight);
        self.proposal_log_probability
            .extend(other.proposal_log_probability);
        self.importance_log_weight
            .extend(other.importance_log_weight);
    }
}

pub fn to_polars_dataframe(py: Python<'_>, output: OutputTable) -> PyResult<PyObject> {
    let polars = py.import("polars")?;
    let data = PyDict::new(py);
    data.set_item("context_id", PyList::new(py, output.context_id)?)?;
    data.set_item("draw_id", PyList::new(py, output.draw_id)?)?;
    data.set_item("layer", PyList::new(py, output.layer)?)?;
    data.set_item("origin", PyList::new(py, output.origin)?)?;
    data.set_item("destination", PyList::new(py, output.destination)?)?;
    data.set_item(
        "local_log_weight",
        PyList::new(py, output.local_log_weight)?,
    )?;
    data.set_item(
        "total_log_weight",
        PyList::new(py, output.total_log_weight)?,
    )?;
    data.set_item(
        "proposal_log_probability",
        PyList::new(py, output.proposal_log_probability)?,
    )?;
    data.set_item(
        "importance_log_weight",
        PyList::new(py, output.importance_log_weight)?,
    )?;
    Ok(polars.getattr("DataFrame")?.call1((data,))?.into())
}
