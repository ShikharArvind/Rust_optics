use ndarray::prelude::*;
use ndarray::{concatenate, Slice};
use ndrustfft::*;
use num_complex::Complex64;

pub fn fftshift<T: std::clone::Clone>(x: &ArrayD<T>) -> ArrayD<T> {
    //General FFTSHIFT function for n dimensions, required ArrayD hence array_input.into_dyn() necessary

    // For first Axis i.e. Axis(0)
    let mut n = x.len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway
    let mut n_div_2 = (f64::from(n) / 2.0).ceil() as i32; // Ceil of n/2 works for both odd and even n
    let mut slice_1 = Slice::from(n_div_2..n);
    let mut slice_2 = Slice::from(0..n_div_2);
    let mut arr = concatenate![
        Axis(0),
        x.slice_axis(Axis(0), slice_1),
        x.slice_axis(Axis(0), slice_2)
    ];
    // For rest of the Axes
    if x.ndim() >= 1 {
        for i in 1..x.ndim() {
            n = x.len_of(Axis(i)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway
            n_div_2 = (f64::from(n) / 2.0).ceil() as i32; // Ceil of n/2 works for both odd and even n
            slice_1 = Slice::from(n_div_2..n);
            slice_2 = Slice::from(0..n_div_2);
            arr = concatenate![
                Axis(i),
                arr.slice_axis(Axis(i), slice_1),
                arr.slice_axis(Axis(i), slice_2)
            ]
        }
    }

    return arr;
}

pub fn ifftshift<T: std::clone::Clone>(x: &ArrayD<T>) -> ArrayD<T> {
    //General IFFTSHIFT function for n dimensions, required ArrayD hence &(array_input.into_dyn()) necessary

    // For first Axis i.e. Axis(0)
    let mut n = x.len_of(Axis(0)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway
    let mut n_div_2 = (f64::from(n) / 2.0).floor() as i32; // Floor of n/2 works for both odd and even n
    let mut slice_1 = Slice::from(n_div_2..n);
    let mut slice_2 = Slice::from(0..n_div_2);
    let mut arr = concatenate![
        Axis(0),
        x.slice_axis(Axis(0), slice_1),
        x.slice_axis(Axis(0), slice_2)
    ];
    // For rest of the Axes
    if x.ndim() >= 1 {
        for i in 1..x.ndim() {
            n = x.len_of(Axis(i)) as i32; // Casting usize (u64) to i32 , array sizes are limited to isize::MAX anyway
            n_div_2 = (f64::from(n) / 2.0).floor() as i32; // Floor of n/2 works for both odd and even n
            slice_1 = Slice::from(n_div_2..n);
            slice_2 = Slice::from(0..n_div_2);
            arr = concatenate![
                Axis(i),
                arr.slice_axis(Axis(i), slice_1),
                arr.slice_axis(Axis(i), slice_2)
            ]
        }
    }
    return arr;
}

pub fn fftn(input: &ArrayD<Complex64>) -> ArrayD<Complex64> {
    //For now its only 2D but plan on implementing for n dimensions hence ArrayD used
    let mut ft_along_axis_0 = ArrayD::<Complex64>::zeros(input.raw_dim());
    let mut handler = FftHandler::new(ft_along_axis_0.len_of(Axis(0)));
    ndfft(&input, &mut ft_along_axis_0, &mut handler, 0);
    handler = FftHandler::new(ft_along_axis_0.len_of(Axis(1)));
    let mut ft_along_axis_1_0 = ArrayD::<Complex64>::zeros(ft_along_axis_0.raw_dim());
    ndfft(&ft_along_axis_0, &mut ft_along_axis_1_0, &mut handler, 1);
    return ft_along_axis_1_0;
}

pub fn ifftn(input: &ArrayD<Complex64>) -> ArrayD<Complex64> {
    //For now its only 2D but plan on implementing for n dimensions hence ArrayD used
    let mut ift_along_axis_0 = ArrayD::<Complex64>::zeros(input.raw_dim());
    let mut handler = FftHandler::new(ift_along_axis_0.len_of(Axis(0)));
    ndifft(&input, &mut ift_along_axis_0, &mut handler, 0);
    handler = FftHandler::new(ift_along_axis_0.len_of(Axis(1)));
    let mut ift_along_axis_1_0 = ArrayD::<Complex64>::zeros(ift_along_axis_0.raw_dim());
    ndfft(&ift_along_axis_0, &mut ift_along_axis_1_0, &mut handler, 1);
    return ift_along_axis_1_0;
}
