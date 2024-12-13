use rayon::prelude::*;

pub trait MathUtil {
    fn quantile(&self, q: &[f64]) -> Vec<f64>;
}

impl MathUtil for [f64] {
    fn quantile(&self, q: &[f64]) -> Vec<f64> {
        let n = self.len() as f64;
        let mut sorted = self.to_vec();
        sorted.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        q.iter()
            .map(|&quantile| {
                let m = 1.0 - quantile;
                let pos = quantile * n + m - 1.0;
                let j = pos.floor().max(0.0) as usize;
                let g = pos.fract();
                if j + 1 < sorted.len() {
                    (1.0 - g) * sorted[j] + g * sorted[j + 1]
                } else {
                    sorted[j]
                }
            })
            .collect()
    }
}

#[inline(always)]
pub fn calculate_uplift(before: f64, after: f64) -> f64 {
    (after - before) / before
}