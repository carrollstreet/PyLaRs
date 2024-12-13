use crate::tools::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use pyo3::prelude::*;

#[pyfunction(
    signature = (
        args,
        confidence_level = 0.95, 
        n_resamples = 10_000, 
        two_sided = true,
    )
)]
#[pyo3(text_signature = "(args, confidence_level=0.95, n_resamples=10000, two_sided=True)")]
/// """
/// Performs a permutation test to evaluate the statistical significance of the difference in means
/// (or mean ratios) between two or four sets of samples.
///
/// Args:
///     args (List[List[float]]): A list containing either two or four lists of floats.
///         - If two lists are provided: They represent two samples for comparison.
///           The function will test the difference in their means.
///         - If four lists are provided: They represent two pairs of (numerator, denominator) data sets.
///           The function will test the difference in their mean ratios (sum(num)/sum(den) for each pair).
///     confidence_level (float, optional): The confidence level for constructing the confidence interval.
///         Default is 0.95.
///     n_resamples (int, optional): The number of permutation resamples to generate for building the null distribution.
///         Default is 10000.
///     two_sided (bool, optional): If True, returns a two-sided p-value. If False, returns a one-sided p-value.
///         Default is True.
///
/// Returns:
///     Tuple[float, float, float, (float, float)]:
///         A tuple containing:
///         - p_value (float): The p-value reflecting the probability of obtaining a result at least as extreme
///           as the observed difference under the null hypothesis.
///         - uplift (float): The relative difference (observed_diff / baseline_mean), where baseline_mean is the mean
///           (or ratio) of the first sample/pair.
///         - observed_diff (float): The observed absolute difference in means or mean ratios (e.g., mean_2 - mean_1).
///         - (float, float): The confidence interval bounds for the observed difference based on the specified confidence level.
/// """
pub fn permutation_test(
    args: Vec<Vec<f64>>,
    confidence_level: f64,
    n_resamples: u64,
    two_sided: bool,
) -> (f64, f64, f64, (f64, f64)) {
    let left_q = (1.0 - confidence_level) / 2.0;
    let right_q = 1.0 - left_q;

    let (vec_diffs, uplift, observed_diff): (Vec<f64>, f64, f64) = match args.len() {
        2 => {
            let (len_a, len_b) = (args[0].len(), args[1].len());
            let mut combined: Vec<f64> = Vec::with_capacity(len_a + len_b);
            combined.extend_from_slice(&args[0]);
            combined.extend_from_slice(&args[1]);
            let len_comb = combined.len();
            let (a_mean, b_mean) = (
                args[0].iter().sum::<f64>() / len_a as f64,
                args[1].iter().sum::<f64>() / len_b as f64,
            );

            let observed_diff = b_mean - a_mean;
            let uplift = observed_diff / a_mean;

            let vec_diffs: Vec<f64> = (0..n_resamples)
                .into_par_iter()
                .map(|i| {
                    let seed: u64 = i ^ i.wrapping_mul(0x9e3779b97f4a7c15);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let mut ids: Vec<usize> = (0..len_comb).collect();
                    ids.shuffle(&mut rng);

                    let sum_a: f64 = ids[..len_a]
                        .iter()
                        .map(|id| unsafe { combined.get_unchecked(*id) })
                        .sum();
                    let sum_b: f64 = ids[len_a..]
                        .iter()
                        .map(|id| unsafe { combined.get_unchecked(*id) })
                        .sum();
                    (sum_b / len_b as f64) - (sum_a / len_a as f64)
                })
                .collect();

            (vec_diffs, uplift, observed_diff)
        }
        4 => {
            let (len_a, len_b) = (args[0].len(), args[2].len());

            if len_a != args[1].len() || len_b != args[3].len() {
                panic!("Each pair of arrays must be of equal length.");
            }

            let (ratio_a, ratio_b) = (
                args[0].iter().sum::<f64>() / args[1].iter().sum::<f64>(),
                args[2].iter().sum::<f64>() / args[3].iter().sum::<f64>(),
            );

            let observed_diff = ratio_b - ratio_a;
            let uplift = observed_diff / ratio_a;

            let mut numerators = Vec::with_capacity(len_a + len_b);
            let mut denominators = Vec::with_capacity(len_a + len_b);

            numerators.extend_from_slice(&args[0]);
            denominators.extend_from_slice(&args[1]);
            numerators.extend_from_slice(&args[2]);
            denominators.extend_from_slice(&args[3]);

            let len_comb = numerators.len();

            let vec_diffs: Vec<f64> = (0..n_resamples)
                .into_par_iter()
                .map(|i| {
                    let seed: u64 = i ^ i.wrapping_mul(0x9e3779b97f4a7c15);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let mut ids: Vec<usize> = (0..len_comb).collect();
                    ids.shuffle(&mut rng);

                    let (sum_a_num, sum_a_den): (f64, f64) = ids[..len_a]
                        .iter()
                        .map(|&id| unsafe {
                            (numerators.get_unchecked(id), denominators.get_unchecked(id))
                        })
                        .fold((0.0, 0.0), |(num, den), (a, b)| (num + a, den + b));

                    let (sum_b_num, sum_b_den): (f64, f64) = ids[len_a..]
                        .iter()
                        .map(|&id| unsafe {
                            (numerators.get_unchecked(id), denominators.get_unchecked(id))
                        })
                        .fold((0.0, 0.0), |(num, den), (a, b)| (num + a, den + b));

                    (sum_b_num / sum_b_den) - (sum_a_num / sum_a_den)
                })
                .collect();

            (vec_diffs, uplift, observed_diff)
        }
        _ => {
            panic!("Input must contain either 2 or 4 vectors.");
        }
    };
    let p = (vec_diffs.iter().filter(|i| observed_diff > **i).count() + 1) as f64
        / (n_resamples + 1) as f64;
    let p_value = (2.0 - 2.0 * p).min(p * 2.0);
    let q = vec_diffs.quantile(&[left_q, right_q]);
    (
        if two_sided { p_value } else { p },
        uplift,
        observed_diff,
        (q[0], q[1]),
    )
}
