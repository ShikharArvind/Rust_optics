use ndarray::concatenate;
use ndarray::prelude::*;
use ndrustfft::*;

// Implement simple fftshift for 1d using slices and concatenation
pub fn fftshift1d(x: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let n = (*x).len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
    let n_div_2 = (f64::from(n) / 2.0).ceil() as i32; // Ceil of n/2 works for both odd and even n
    concatenate![
        Axis(0),
        (*x).slice(s![n_div_2..n]),
        (*x).slice(s![0..n_div_2])
    ]
}

// Implement simple ifftshift for 1d using slices and concatenation
pub fn ifftshift1d(x: &Array1<Complex<f64>>) -> Array1<Complex<f64>> {
    let n = (*x).len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
    let n_div_2 = (f64::from(n) / 2.0).floor() as i32; // Ceil of n/2 works for both odd and even n
    concatenate![
        Axis(0),
        (*x).slice(s![n_div_2..n]),
        (*x).slice(s![0..n_div_2])
    ]
}

// Implement simple fftshift for 2d using slices and concatenation. This is very much an extension
// of 1D swapping. Honestly a common fftshift and iffshit for 1d->3d can be implemented like numpy
// Implement a generic type i.e. Complex<f64> or <f64> (TODO)
pub fn fftshift2d(x: &Array2<f64>) -> Array2<f64> {
    let n = (*x).len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
    let n_div_2 = (f64::from(n) / 2.0).ceil() as i32; // Ceil of n/2 works for both odd and even n
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

// Implement simple ifftshift for 2d using slices and concatenation.
pub fn ifftshift2d(x: &Array2<f64>) -> Array2<f64> {
    let n = (*x).len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
    let n_div_2 = (f64::from(n) / 2.0).floor() as i32; // Floor of n/2 works for both odd and even n
    let concat_ = concatenate![
        Axis(0),
        (*x).slice(s![n_div_2..n, ..]),
        (*x).slice(s![0..n_div_2, ..])
    ];
    let m = (*x).len_of(Axis(1)) as i32;
    let m_div_2 = (f64::from(m) / 2.0).floor() as i32;
    concatenate![
        Axis(1),
        concat_.slice(s![.., m_div_2..m]),
        concat_.slice(s![.., 0..m_div_2])
    ]
}
