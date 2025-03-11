mod perm;
mod tools;
mod binom_coef;
mod bootstrapping;

use binom_coef::*;
use perm::*;
use bootstrapping::*;
use pyo3::prelude::*;

#[pymodule]
fn pylars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(permutation_test, m)?)?;
    m.add_function(wrap_pyfunction!(binom, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_vec, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(stratified_bootstrap, m)?)?;
    Ok(())
}


