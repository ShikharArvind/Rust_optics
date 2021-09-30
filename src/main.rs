// Learning Rust and Fourier Optics :)
// Refer Computational Fourier Optics by Davis Volez
use ndarray::prelude::*;
use num_complex::Complex64;
use std::fs::File;
use std::io::*;
pub mod apertures;
pub mod fftshift;
pub mod meshgrid;
pub mod propagation;

fn main() {
    //Simple rectangular 1d function and it Fourier transform
    // let w: f64 = 0.055;
    // let l: f64 = 2.0;
    // let m: f64 = 201.0;
    // let dx: f64 = l / m;
    // let x_range = Array::range(-l / 2.0, l / 2.0, dx);
    // let array_input = x_range.mapv(|a| apertures::rect_1d(a / (2.0 * w)));
    // let fftshift_array_input = fftshift::fftshiftgen(&array_input.into_dyn());
    // let mut handler: FftHandler<f64> = FftHandler::new(fftshift_array_input.len());
    // let mut fft_out = Array1::<Complex<f64>>::zeros(fftshift_array_input.len()).into_dyn();
    // ndfft_par(&fftshift_array_input, &mut fft_out, &mut handler, 0);
    // fft_out = fftshift::ifftshiftgen(&fft_out);
    // println!("{} ", fft_out);
    //
    // //Test code odd odd and even size square arrays for the fftshift in 2d
    // let array2d_odd = arr2(&[
    //     [0.0, 1.0, 2.0, 3.0, 4.0],
    //     [5.0, 6.0, 7.0, 8.0, 9.0],
    //     [10.0, 11.0, 12.0, 13.0, 14.0],
    //     [15.0, 16.0, 17.0, 18.0, 19.0],
    //     [20.0, 21.0, 22.0, 23.0, 24.0],
    // ]);
    // let array2d_eve = arr2(&[
    //     [0.0, 1.0, 2.0, 3.0],
    //     [5.0, 6.0, 7.0, 8.0],
    //     [10.0, 11.0, 12.0, 13.0],
    //     [15.0, 16.0, 17.0, 18.0],
    // ]);
    // let fft2d_test = fftshift::fftshiftgen(&array2d_odd.into_dyn());
    // println!("{}", fft2d_test);
    // println!("{}", fftshift::ifftshiftgen(&fft2d_test));

    // let array_x = arr1(&[1.0, 2.0, 3.0]);
    // let array_y = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    // let (arr_test_x, arr_test_y) = meshgrid::meshgrid(&array_x, &array_y);
    //
    // println!("{}", arr_test_x);
    // println!("{}", arr_test_y);

    //Fresnel Propagation of square (Chapter 5, Section 5.1)
    let w: f64 = 0.011;
    let z: f64 = 2000.0;
    let l: f64 = 0.5;
    let m: f64 = 250.0;
    let dx: f64 = l / m;
    let lam: f64 = 0.5e-06;
    let x1 = Array::range(-l / 2.0, l / 2.0, dx);
    let (X1, Y1) = meshgrid::meshgrid(&x1, &x1);
    let u1 = X1.mapv(|x| apertures::rect_1d(x / (2.0 * w)))
        * Y1.mapv(|y| apertures::rect_1d(y / (2.0 * w)));
    let mut u2 = propagation::propTF(&u1, &l, &lam, &z);
    u2.map_inplace(|a| nan_to_zero(a)); // Replace NaN with zeros

    //Print u2 out to file so that numpy.genfromtxt can read
    let mut f = File::create("output.txt").expect("Unable to create file");
    write!(f, "[");
    for i in 0..u2.len_of(Axis(0)) {
        write!(f, "[");
        for iter in u2.index_axis(Axis(1), i).indexed_iter() {
            if iter.0 == u2.len_of(Axis(1)) - 1 {
                write!(f, "{}", *(iter.1));
            } else {
                write!(f, "{},", *(iter.1));
            }
        }
        if i == u2.len_of(Axis(0)) - 1 {
            write!(f, "]");
        } else {
            writeln!(f, "]");
        }
    }
    write!(f, "]");
}

fn nan_to_zero(x: &mut Complex64) {
    if (*x).re.is_nan() {
        (*x).re = 0.0;
    }
    if (*x).im.is_nan() {
        (*x).im = 0.0;
    }
}
