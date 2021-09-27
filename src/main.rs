// Learning Rust and Fourier Optics :)
// Refer Computational Fourier Optics by Davis Volez - Chap 3
use ndarray::prelude::*;
use ndrustfft::*;
mod apertures;
mod fftshift;
fn main() {
    //Simple rectangular 1d function and it Fourier transform
    let w: f64 = 0.055;
    let l: f64 = 2.0;
    let m: f64 = 201.0;
    let dx: f64 = l / m;
    let x_range = Array::range(-l / 2.0, l / 2.0, dx);
    let array_input = x_range.mapv(|a| apertures::rect_1d(a / (2.0 * w)));
    let fftshift_array_input = fftshift::fftshiftgen(&array_input.into_dyn());
    let mut handler: FftHandler<f64> = FftHandler::new(fftshift_array_input.len());
    let mut fft_out = Array1::<Complex<f64>>::zeros(fftshift_array_input.len()).into_dyn();
    ndfft_par(&fftshift_array_input, &mut fft_out, &mut handler, 0);
    fft_out = fftshift::ifftshiftgen(&fft_out);
    println!("{} ", fft_out);

    //Test code odd odd and even size square arrays for the fftshift in 2d
    let array2d_odd = arr2(&[
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [15.0, 16.0, 17.0, 18.0, 19.0],
        [20.0, 21.0, 22.0, 23.0, 24.0],
    ]);
    let array2d_eve = arr2(&[
        [0.0, 1.0, 2.0, 3.0],
        [5.0, 6.0, 7.0, 8.0],
        [10.0, 11.0, 12.0, 13.0],
        [15.0, 16.0, 17.0, 18.0],
    ]);
    let fft2d_test = fftshift::fftshiftgen(&array2d_odd.into_dyn());
    println!("{}", fft2d_test);
    println!("{}", fftshift::ifftshiftgen(&fft2d_test));
}
