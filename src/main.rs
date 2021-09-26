// Learning Rust and Fourier Optics :)
// This is basic implementation of a 1D rectangular function with its Fourier Transform
// Refer Computational Fourier Optics by Davis Volez - Chap 3, section 3.2
use ndarray::concatenate;
use ndarray::prelude::*;
use ndrustfft::*;

fn main() {
    let w: f64 = 0.055;
    let l: f64 = 2.0;
    let m: f64 = 200.0;
    let dx: f64 = l / m;
    let x_range = Array::range(-l / 2.0, l / 2.0, dx);
    let array_input = x_range.mapv(|a| rect_1d(a / (2.0 * w)));
    let fftshift_array_input = fftshift1d(&array_input);
    let mut handler: FftHandler<f64> = FftHandler::new(fftshift_array_input.len());
    let mut fft_out = Array1::<Complex<f64>>::zeros(fftshift_array_input.len());
    ndfft_par(&fftshift_array_input, &mut fft_out, &mut handler, 0);
    println!("{} ", fftshift_array_input);
    println!("{} ", fft_out);
}

// Implement simple fftshift for 1d using slices and concatenation
// TODO : move this outside main.rs as a separate module APERTURE FUNCTIONS (?)
fn rect_1d(x: f64) -> Complex<f64> {
    if x.abs() <= 0.5 {
        Complex::new(1.0, 0.0)
    } else {
        Complex::new(0.0, 0.0)
    }
}

// Implement simple fftshift for 1d using slices and concatenation
// Panic when size of input array is 1 or 0 as it is catastrophic, Result<E> would not help (?) I guess
// TODO : move this outside main.rs as a separate module FFTSHIT for 1d and 2d (?)
fn fftshift1d(x: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let n = (*x).len();
    if n > 1 {
        if n % 2 == 0 {
            concatenate![Axis(0), (*x).slice(s![n / 2..n]), (*x).slice(s![0..n / 2])]
        } else {
            concatenate![
                Axis(0),
                (*x).slice(s![(n + 1) / 2..n]),
                (*x).slice(s![0..(n + 1) / 2])
            ]
        }
    } else {
        panic!("Input array size/len is less than 2")
    }
}
