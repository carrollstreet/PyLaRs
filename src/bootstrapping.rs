use crate::tools::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use pyo3::prelude::*;
use std::cmp::Ordering;

#[pyfunction(signature = (vec, n_resamples = 10_000))]
#[pyo3(text_signature = "(vec, n_resamples=10000)")]
/// """
/// Performs bootstrap resampling on a vector of floating-point numbers, returning a distribution of sample means.
///
/// Args:
///     vec (List[float]): The input vector of floats.
///     n_resamples (int, optional): The number of bootstrap resamples. Default is 10000.
///
/// Returns:
///     List[float]: A list of bootstrap sample means.
/// """
pub fn bootstrap_vec(vec: Vec<f64>, n_resamples: u64) -> Vec<f64> {
    let len_vec = vec.len();
    let dist = rand::distributions::Uniform::new(0, len_vec);

    (0..n_resamples)
        .into_par_iter()
        .map(|i| {
            let seed: u64 = i ^ i.wrapping_mul(0x9e3779b97f4a7c15);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let mut sum = 0.0;
            for _ in 0..len_vec {
                let idx = dist.sample(&mut rng);
                unsafe {
                    sum += *vec.get_unchecked(idx);
                }
            }
            sum / len_vec as f64
        })
        .collect()
}

#[pyfunction(signature = (args, confidence_level = 0.95, n_resamples = 10_000, ind = true, two_sided = true))]
#[pyo3(text_signature = "(args, confidence_level=0.95, n_resamples=10000, ind=True, two_sided=True)")]
/// """
/// Performs a bootstrap analysis to evaluate the statistical significance of the difference in means 
/// (or mean ratios) between two or four sets of samples.
///
/// Args:
///     args (List[List[float]]): A list containing either two or four lists of floats.
///         If two are provided, they represent two independent samples to compare.
///         If four are provided, they represent two pairs of (numerator, denominator) data to compare ratios.
///     confidence_level (float, optional): The confidence level for the interval. Default is 0.95.
///     n_resamples (int, optional): The number of bootstrap resamples. Default is 10000.
///     ind (bool, optional): If True, samples are treated as independent. If False, samples are treated as paired. Default is True.
///     two_sided (bool, optional): If True, computes a two-sided p-value. Otherwise, one-sided. Default is True.
///
/// Returns:
///     Tuple[float, float, float, float, (float, float)]:
///         A tuple containing:
///         - p_value (float): The p-value for the test (two-sided or one-sided depending on `two_sided`).
///         - mean_1 (float): The mean (or ratio) of the first dataset.
///         - mean_2 (float): The mean (or ratio) of the second dataset.
///         - uplift (float): The observed difference in means or ratios (mean_2 - mean_1).
///         - (float, float): The confidence interval bounds for the difference.
/// """
pub fn bootstrap(
    args: Vec<Vec<f64>>,
    confidence_level: f64,
    n_resamples: u64,
    ind: bool,
    two_sided: bool,
) -> (f64, f64, f64, f64, (f64, f64)) {
    let left_q = (1.0 - confidence_level) / 2.0;
    let right_q = 1.0 - left_q;
    let (uplift_diffs, mean_1, mean_2, uplift): (Vec<f64>, f64, f64, f64) = match args.len() {
        2 => {
            let len_vec_1 = args[0].len();
            let len_vec_2 = args[1].len();
            if !ind && len_vec_1 != len_vec_2 {
                panic!("For non ind test all arrays must have same size")
            }
            let (mean_1, mean_2): (f64, f64) = (
                args[0].iter().sum::<f64>() / len_vec_1 as f64,
                args[1].iter().sum::<f64>() / len_vec_2 as f64,
            );
            let uplift = calculate_uplift(mean_1, mean_2);
            let min_len = len_vec_1.min(len_vec_2);
            let dist_1 = rand::distributions::Uniform::new(0, len_vec_1);
            let dist_2 = rand::distributions::Uniform::new(0, len_vec_2);

            let uplift_diffs: Vec<f64> = (0..n_resamples)
                .into_par_iter()
                .map(|i| {
                    let seed: u64 = i ^ i.wrapping_mul(0x9e3779b97f4a7c15);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

                    let mut sum_vec_1 = 0.0;
                    let mut sum_vec_2 = 0.0;
                    if ind {
                        for _ in 0..min_len {
                            let idx_1 = dist_1.sample(&mut rng);
                            let idx_2 = dist_2.sample(&mut rng);
                            unsafe {
                                sum_vec_1 += *args[0].get_unchecked(idx_1);
                                sum_vec_2 += *args[1].get_unchecked(idx_2);
                            }
                        }
                        match len_vec_1.cmp(&len_vec_2) {
                            Ordering::Greater => {
                                for _ in 0..(len_vec_1 - len_vec_2) {
                                    let idx_1 = dist_1.sample(&mut rng);
                                    unsafe {
                                        sum_vec_1 += *args[0].get_unchecked(idx_1);
                                    }
                                }
                            }
                            Ordering::Less => {
                                for _ in 0..(len_vec_2 - len_vec_1) {
                                    let idx_2 = dist_2.sample(&mut rng);
                                    unsafe {
                                        sum_vec_2 += *args[1].get_unchecked(idx_2);
                                    }
                                }
                            }
                            Ordering::Equal => {}
                        }
                    } else {
                        for _ in 0..min_len {
                            let idx_1 = dist_1.sample(&mut rng);
                            unsafe {
                                sum_vec_1 += *args[0].get_unchecked(idx_1);
                                sum_vec_2 += *args[1].get_unchecked(idx_1);
                            }
                        }
                    }
                    let mean_1 = sum_vec_1 / len_vec_1 as f64;
                    let mean_2 = sum_vec_2 / len_vec_2 as f64;
                    calculate_uplift(mean_1, mean_2)
                })
                .collect();
            (uplift_diffs, mean_1, mean_2, uplift)
        }
        4 => {
            let vec_sizes: Vec<usize> = args.iter().map(|vec| vec.len()).collect();
            if !ind {
                if !(vec_sizes[0] == vec_sizes[1]
                    && vec_sizes[2] == vec_sizes[3]
                    && vec_sizes[0] == vec_sizes[2])
                {
                    panic!("For non ind test all arrays must have same size")
                }
            } else if vec_sizes[0] != vec_sizes[1] || vec_sizes[2] != vec_sizes[3] {
                panic!("Each pair of arrays must be of equal length.");
            }
            let (mean_1, mean_2): (f64, f64) = (
                args[0].iter().sum::<f64>() / args[1].iter().sum::<f64>(),
                args[2].iter().sum::<f64>() / args[3].iter().sum::<f64>(),
            );
            let uplift = calculate_uplift(mean_1, mean_2);
            let dist_1 = rand::distributions::Uniform::new(0, vec_sizes[0]);
            let dist_2 = rand::distributions::Uniform::new(0, vec_sizes[2]);
            let min_len = vec_sizes[0].min(vec_sizes[2]);
            let uplift_diffs: Vec<f64> = (0..n_resamples)
                .into_par_iter()
                .map(|i| {
                    let seed: u64 = i ^ i.wrapping_mul(0x9e3779b97f4a7c15);
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

                    let mut sum_num_1 = 0.0;
                    let mut sum_denum_1 = 0.0;
                    let mut sum_num_2 = 0.0;
                    let mut sum_denum_2 = 0.0;
                    if ind {
                        for _ in 0..min_len {
                            let idx_1 = dist_1.sample(&mut rng);
                            let idx_2 = dist_2.sample(&mut rng);
                            unsafe {
                                sum_num_1 += *args[0].get_unchecked(idx_1);
                                sum_denum_1 += *args[1].get_unchecked(idx_1);
                                sum_num_2 += *args[2].get_unchecked(idx_2);
                                sum_denum_2 += *args[3].get_unchecked(idx_2);
                            }
                        }
                        match vec_sizes[0].cmp(&vec_sizes[2]) {
                            Ordering::Greater => {
                                for _ in 0..(vec_sizes[0] - vec_sizes[2]) {
                                    let idx_1 = dist_1.sample(&mut rng);
                                    unsafe {
                                        sum_num_1 += *args[0].get_unchecked(idx_1);
                                        sum_denum_1 += *args[1].get_unchecked(idx_1);
                                    }
                                }
                            }
                            Ordering::Less => {
                                for _ in 0..(vec_sizes[2] - vec_sizes[0]) {
                                    let idx_2 = dist_2.sample(&mut rng);
                                    unsafe {
                                        sum_num_2 += *args[2].get_unchecked(idx_2);
                                        sum_denum_2 += *args[3].get_unchecked(idx_2);
                                    }
                                }
                            }
                            Ordering::Equal => {}
                        }
                    } else {
                        for _ in 0..min_len {
                            let idx_1 = dist_1.sample(&mut rng);
                            unsafe {
                                sum_num_1 += *args[0].get_unchecked(idx_1);
                                sum_denum_1 += *args[1].get_unchecked(idx_1);
                                sum_num_2 += *args[2].get_unchecked(idx_1);
                                sum_denum_2 += *args[3].get_unchecked(idx_1);
                            }
                        }
                    }
                    let mean_1 = sum_num_1 / sum_denum_1;
                    let mean_2 = sum_num_2 / sum_denum_2;
                    calculate_uplift(mean_1, mean_2)
                })
                .collect();

            (uplift_diffs, mean_1, mean_2, uplift)
        }
        _ => {
            panic!("Input must contain either 2 or 4 vectors.");
        }
    };
    let p: f64 =
        (uplift_diffs.iter().filter(|&&i| i > 0.0).count() as f64 + 1.0) / (n_resamples + 1) as f64;
    let p_value = (2.0 - 2.0 * p).min(p * 2.0);
    let q = uplift_diffs.quantile(&[left_q, right_q]);
    (
        if two_sided { p_value } else { p },
        mean_1,
        mean_2,
        uplift,
        (q[0], q[1]),
    )
}