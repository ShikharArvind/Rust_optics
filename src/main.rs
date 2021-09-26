// Learning Rust and Fourier Optics :)
// This is basic implementation of a 1D rectangular function with its Fourier Transform
// Refer Computational Fourier Optics by Davis Volez - Chap 3, section 3.2
use ndarray::arr2;
use ndarray::concatenate;
use ndarray::prelude::*;
use ndrustfft::*;

fn main() {
    // let w: f64 = 0.055;
    // let l: f64 = 2.0;
    // let m: f64 = 201.0;
    // let dx: f64 = l / m;
    // let x_range = Array::range(-l / 2.0, l / 2.0, dx);
    // let array_input = x_range.mapv(|a| rect_1d(a / (2.0 * w)));
    // let fftshift_array_input = fftshift1d(&array_input);
    // let mut handler: FftHandler<f64> = FftHandler::new(fftshift_array_input.len());
    // let mut fft_out = Array1::<Complex<f64>>::zeros(fftshift_array_input.len());
    // ndfft_par(&fftshift_array_input, &mut fft_out, &mut handler, 0);
    // fft_out = ifftshift1d(&fft_out);
    // println!("{} ", fft_out);

    //Test code for the fftshift2d
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

    println!("{}", array2d_eve);
    println!("{}", fftshift2d(&array2d_eve));
    println!("{}", array2d_odd);
    println!("{}", fftshift2d(&array2d_odd));
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
// TODO : move this outside main.rs as a separate module like FTSHIFT for 1d and 2d (?)
// TODO : skip divide by 2 check for odd or even and implement ceil like 2D
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

// Implement simple ifftshift for 1d using slices and concatenation
// Panic when size of input array is 1 or 0 as it is catastrophic, Result<E> would not help (?) I guess
// TODO : move this outside main.rs as a separate module like FTSHIFT for 1d and 2d (?)
fn ifftshift1d(x: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let n = (*x).len();
    if n > 1 {
        if n % 2 == 0 {
            concatenate![Axis(0), (*x).slice(s![n / 2..n]), (*x).slice(s![0..n / 2])]
        } else {
            concatenate![
                Axis(0),
                (*x).slice(s![(n - 1) / 2..n]),
                (*x).slice(s![0..(n - 1) / 2])
            ]
        }
    } else {
        panic!("Input array size/len is less than 2")
    }
}

// Implement simple fftshift for 2d using slices and concatenation. This is very much an extension
// of 1D swapping. Honestly a common fftshift and iffshit for 1d->3d can be implemented like numpy
// Panic / Edge cases not considered yet (TODO)
// TODO : move this outside main.rs as a separate module like FTSHIFT for 1d and 2d (?)

fn fftshift2d(x: &Array2<f64>) -> Array2<f64> {
    // Is it possible to implement a generic type i.e. Complex<f64> or <f64> (TODO) ???
    let n = (*x).len_of(Axis(0)) as i32; // Not sure if casting usize as i32 has edge cases, need to check (TODO)
    let n_div_2 = (f64::from(n) / 2.0).ceil() as i32; // Ceil of n/2 for both odd and even n
    let concat_ = concatenate![
        Axis(0),
        (*x).slice(s![n_div_2..n, ..]),
        (*x).slice(s![0..n_div_2, ..])
    ];
    let m = (*x).len_of(Axis(1)) as i32;
    let m_div_2 = (f64::from(m) / 2.0).ceil() as i32;
    concatenate![
        Axis(1),
        concat_.slice(s![.., m_div_2..m]),
        concat_.slice(s![.., 0..m_div_2])
    ]
}
