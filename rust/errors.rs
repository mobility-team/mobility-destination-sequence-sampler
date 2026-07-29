use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SamplerError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("python interop error: {0}")]
    Python(String),
    #[error("context {context_id} has no feasible destination sequence from zone {origin}")]
    NoFeasibleSequence { context_id: u64, origin: u32 },
    #[error(
        "bounded search found no feasible destination sequence for context {context_id} from zone {origin}; this does not prove the context is infeasible"
    )]
    BoundedSearchNoPlan { context_id: u64, origin: u32 },
}

impl From<PyErr> for SamplerError {
    fn from(value: PyErr) -> Self {
        Self::Python(value.to_string())
    }
}

impl From<SamplerError> for PyErr {
    fn from(value: SamplerError) -> Self {
        PyValueError::new_err(value.to_string())
    }
}
