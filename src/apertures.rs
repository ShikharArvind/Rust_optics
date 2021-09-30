use num_complex::Complex64;

// Simple rect function
pub fn rect_1d(x: f64) -> Complex64 {
    if x.abs() <= 0.5 {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    }
}
