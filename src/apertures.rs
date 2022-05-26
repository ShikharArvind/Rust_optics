use ndarray::prelude::*;
use num_complex::Complex64;
use std::f64::consts::PI;
// Simple rect function
pub fn rect_1d(x: f64) -> Complex64 {
    if x.abs() <= 0.5 {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    }
}

//Implement sinusoidal grating function
pub fn grating(x1: &Array2<f64>, y1: &Array2<f64>, w: &f64, p: &f64) -> Array2<Complex64> {
    x1.mapv(|x| rect_1d(x / w))
        * y1.mapv(|y| rect_1d(y / w))
        * x1.mapv(|x| (0.5 * (1.0 - (2.0 * PI * x / p).cos())))
}
