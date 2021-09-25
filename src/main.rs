// Learning Rust and Fourier Optics :)
// This is basic implementation of a 1D rectangular function with its Fourier Transform
// Refer Computational Fourier Optics by Davis Volez - Chap 3, section 3.2
use ndarray::prelude::*;
use ndrustfft::*;

fn main() {
    let w: f64 = 0.055;
    let l: f64 = 2.0;
    let m: f64 = 200.0;
    let dx: f64 = l / m;
    let x_range = Array::range(-l / 2.0, l / 2.0, dx);
    let array_input = x_range.map(|a| rect_1d(a / (2.0 * w)));
    let mut handler: FftHandler<f64> = FftHandler::new(array_input.len());
    let mut fft_out = Array1::<Complex<f64>>::zeros(array_input.len());
    ndfft_par(&array_input, &mut fft_out, &mut handler, 0);
    println!("{}", fft_out);
}

fn rect_1d(x: f64) -> Complex<f64> {
    if x.abs() <= 0.5 {
        Complex::new(1.0, 0.0)
    } else {
        Complex::new(0.0, 0.0)
    }
}
