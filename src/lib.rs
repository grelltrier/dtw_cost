#[cfg(feature = "ndarrays")]
pub mod ndarrays;

// Calculate the SQUARED L2 distance (euclidian distance) between two Vec
// The two Vec MUST be of the same length
pub fn sq_l2_dist_vec<T1, T2>(x: &Vec<T1>, y: &Vec<T1>) -> f64
where
    T1: std::ops::Sub<Output = T2> + Copy,
    T2: std::ops::Mul<Output = f64>,
{
    let mut dist = 0.0;
    for idx in 0..x.len() {
        dist += (x[idx] - y[idx]) * (x[idx] - y[idx]);
    }
    dist
}

// Calculate the L2 distance (euclidian distance) between two Vec
// The two Vec MUST be of the same length
pub fn l2_dist_vec<T1, T2>(x: &Vec<T1>, y: &Vec<T1>) -> f64
where
    T1: std::ops::Sub<Output = T2> + Copy,
    T2: std::ops::Mul<Output = f64>,
{
    f64::sqrt(sq_l2_dist_vec(x, y))
}

// Calculate the L2 distance (euclidian distance) for f64
pub fn sq_l2_dist_f64(x: &f64, y: &f64) -> f64 {
    (x - y).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sq_l2_dist_1d_test() {
        assert_eq!(sq_l2_dist_f64(&0.0, &2.0), 4.0);
        assert_eq!(sq_l2_dist_f64(&1.0, &2.0), 1.0);
        assert_eq!(sq_l2_dist_f64(&2.0, &2.0), 0.0);
        assert_eq!(sq_l2_dist_f64(&-0.0, &-2.0), 4.0);
        assert_eq!(sq_l2_dist_f64(&1.0, &-2.0), 9.0);
        assert_eq!(sq_l2_dist_f64(&-0.0, &2.0), 4.0);
    }
}
