use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray::Slice;

// Implement simple fftshift for 1d using slices and concatenation
pub fn fftshift1d<T: std::clone::Clone>(x: &Array1<T>) -> Array1<T> {
    let n = (*x).len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
    let n_div_2 = (f64::from(n) / 2.0).ceil() as i32; // Ceil of n/2 works for both odd and even n
    concatenate![
        Axis(0),
        (*x).slice(s![n_div_2..n]),
        (*x).slice(s![0..n_div_2])
    ]
}

// Implement simple ifftshift for 1d using slices and concatenation
pub fn ifftshift1d<T: std::clone::Clone>(x: &Array1<T>) -> Array1<T> {
    let n = (*x).len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
    let n_div_2 = (f64::from(n) / 2.0).floor() as i32; // Ceil of n/2 works for both odd and even n
    concatenate![
        Axis(0),
        (*x).slice(s![n_div_2..n]),
        (*x).slice(s![0..n_div_2])
    ]
}

// Implement simple fftshift for 2d using slices and concatenation. This is very much an extension
// of 1D swapping.
// A common fftshift and iffshit for 1d->3d can be implemented like numpy (TODO)
pub fn fftshift2d<T: std::clone::Clone>(x: &Array2<T>) -> Array2<T> {
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
pub fn ifftshift2d<T: std::clone::Clone>(x: &Array2<T>) -> Array2<T> {
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

//General FFTSHIFT function for n dimensions, required ArrayD hence array_input.into_dyn() necessary
pub fn fftshiftgen<T: std::clone::Clone>(x: &ArrayD<T>) -> ArrayD<T> {
    let mut concat_ = (*x).clone(); //Is this the best way to do it - Cloning ?
    for i in 0..concat_.ndim() {
        let n = concat_.len_of(Axis(i)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
        let n_div_2 = (f64::from(n) / 2.0).ceil() as i32; // Ceil of n/2 works for both odd and even n
        let slice_1 = Slice::from(n_div_2..n);
        let slice_2 = Slice::from(0..n_div_2);
        concat_ = concatenate![
            Axis(i),
            concat_.slice_axis(Axis(i), slice_1),
            concat_.slice_axis(Axis(i), slice_2)
        ]
    }
    return concat_;
}

//General IFFTSHIFT function for n dimensions, required ArrayD hence &(array_input.into_dyn()) necessary
pub fn ifftshiftgen<T: std::clone::Clone>(x: &ArrayD<T>) -> ArrayD<T> {
    let mut concat_ = (*x).clone(); //Is this the best way to do it - Cloning ?
    for i in 0..concat_.ndim() {
        let n = concat_.len_of(Axis(i)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway (?)
        let n_div_2 = (f64::from(n) / 2.0).floor() as i32; // Ceil of n/2 works for both odd and even n
        let slice_1 = Slice::from(n_div_2..n);
        let slice_2 = Slice::from(0..n_div_2);
        concat_ = concatenate![
            Axis(i),
            concat_.slice_axis(Axis(i), slice_1),
            concat_.slice_axis(Axis(i), slice_2)
        ]
    }
    return concat_;
}
