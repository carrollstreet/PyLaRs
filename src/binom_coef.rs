use pyo3::prelude::*;

#[pyfunction]
pub fn binom(n: u16, k: u16) -> f64
{
    if n == k {
        1.0
    }
    else {
        fn fold(mut start: f64, end: f64) -> f64 {
            let mut mul = 1.0;
            loop {
                mul *= start;
                start += 1.0;
                if start > end {
                    break mul;
                }
            }
        }
        if k > n-k {
            fold(k as f64 + 1.0, n as f64) / fold(1.0, (n - k) as f64)
            }
        else {
            fold((n - k) as f64 + 1.0, n as f64) / fold(1.0, k as f64)
            }
    }
}