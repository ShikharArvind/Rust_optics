use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray::Slice;

//General FFTSHIFT function for n dimensions, required ArrayD hence array_input.into_dyn() necessary
pub fn fftshiftgen<T: std::clone::Clone>(x: &ArrayD<T>) -> ArrayD<T> {
    let mut concat_ = (*x).clone(); //Clone the starting array as mutable and return it
    for i in 0..concat_.ndim() {
        let n = concat_.len_of(Axis(i)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway
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
    let mut concat_ = (*x).clone();
    for i in 0..concat_.ndim() {
        let n = concat_.len_of(Axis(i)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway
        let n_div_2 = (f64::from(n) / 2.0).floor() as i32; // Floor of n/2 works for both odd and even n
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
