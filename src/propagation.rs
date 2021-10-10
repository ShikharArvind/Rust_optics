use super::fft::{fftn, fftshift, ifftn, ifftshift};
use super::meshgrid;
use ndarray::prelude::*;
use num_complex::Complex64;
use std::f64::consts::PI;

pub fn prop_fresnel_transfer_function(
    source: &ArrayD<Complex64>,
    length: &f64,
    wavelength: &f64,
    propagation_distance: &f64,
) -> Array2<Complex64> {
    // source to ArrayD type is not the best as this function is mainly for 2D, but this avoids cloning for fftshift, so not sure if its worth TODO
    //propagation - transfer function H approach
    //assumes same x and y side lengths and
    //uniform sampling -  N = M = number of samples in source
    //length - source and observation plane side length
    //u2 - observation plane field
    //Not explicity dereferncing input params like *source or *length as Rust has auto-dereferncing
    let m: f64 = source.len_of(Axis(1)) as f64;
    let dx: f64 = length / m; // Assumed (dx=dy)
    let fx_range = Array1::<f64>::range(-1.0 / (2.0 * dx), 1.0 / (2.0 * dx), 1.0 / length);
    let k = (2.0 * PI) / wavelength;
    let (mesh_fx, mesh_fy) = meshgrid::meshgrid(&fx_range, &fx_range);

    let transfer_function = (Complex64::i() * k * propagation_distance).exp()
        * (-1.0
            * Complex64::i()
            * PI
            * wavelength
            * propagation_distance
            * (mesh_fx.mapv(|x| Complex64::new(x.powi(2), 0.0))
                + mesh_fy.mapv(|y| Complex64::new(y.powi(2), 0.0))))
        .mapv(|a| a.exp());

    let fft_f1 = fftn(&fftshift(source));
    let ft_u2 = fft_f1 * fftshift(&(transfer_function.into_dyn()));
    return ifftshift(&ifftn(&ft_u2)).into_dimensionality().unwrap();
}

pub fn prop_fresnel_impulse_response(
    source: &ArrayD<Complex64>,
    length: &f64,
    wavelength: &f64,
    propagation_distance: &f64,
) -> Array2<Complex64> {
    // source to ArrayD type is not the best as this function is mainly for 2D, but this avoids cloning for fftshift, so not sure if its worth TODO
    //propagation - impulse response approach
    //assumes same x and y side lengths and
    //uniform sampling -  N = M = number of samples in source
    //length - source and observation plane side length
    //u2 - observation plane field
    //Not explicity dereferncing input params like *source or *length as Rust has auto-dereferncing
    let m: f64 = source.len_of(Axis(1)) as f64;
    let dx: f64 = length / m; // Assumed (dx=dy)
    let x_range = Array1::<f64>::range(-length / 2.0, length / 2.0, dx);
    let k = (2.0 * PI) / wavelength;
    let (mesh_fx, mesh_fy) = meshgrid::meshgrid(&x_range, &x_range);

    let impulse_response = ((Complex64::i() * k * propagation_distance).exp()
        / (Complex64::i() * wavelength * propagation_distance))
        * (((Complex64::i() * k) / (2.0 * propagation_distance))
            * (mesh_fx.mapv(|x| Complex64::new(x.powi(2), 0.0))
                + mesh_fy.mapv(|y| Complex64::new(y.powi(2), 0.0))))
        .mapv(|a| a.exp());
    let transfer_function = fftn(&fftshift(&impulse_response.into_dyn())) * dx.powi(2);
    let fft_f1 = fftn(&fftshift(source));
    let ft_u2 = fft_f1 * transfer_function;
    return ifftshift(&ifftn(&ft_u2)).into_dimensionality().unwrap();
}

pub fn prop_fraunhofer(
    source: &ArrayD<Complex64>,
    length: &f64,
    wavelength: &f64,
    propagation_distance: &f64,
) -> Array2<Complex64> {
    // source to ArrayD type is not the best as this function is mainly for 2D, but this avoids cloning for fftshift, so not sure if its worth TODO
    //propagation - Fraunhofer propagation
    //assumes same x and y side lengths and
    //uniform sampling -  N = M = number of samples in source
    //length1 and length2 - source and observation plane lengths
    //u2 - observation plane field
    //Not explicity dereferncing input params like *source or *length as Rust has auto-dereferncing
    let m: f64 = source.len_of(Axis(1)) as f64;
    let dx1: f64 = length / m; // Assumed (dx=dy)
    let l2: f64 = (wavelength * propagation_distance) / dx1; //obs plane side length
    let dx2: f64 = (wavelength * propagation_distance) / length; //
    let x2 = Array1::<f64>::range(-l2 / 2.0, l2 / 2.0, dx2);
    let k = (2.0 * PI) / wavelength;
    let (mesh_fx, mesh_fy) = meshgrid::meshgrid(&x2, &x2);

    let c = ((Complex64::i() * k * propagation_distance).exp()
        / (Complex64::i() * wavelength * propagation_distance))
        * (((Complex64::i() * k) / (2.0 * propagation_distance))
            * (mesh_fx.mapv(|x| Complex64::new(x.powi(2), 0.0))
                + mesh_fy.mapv(|y| Complex64::new(y.powi(2), 0.0))))
        .mapv(|a| a.exp());
    let fft_f1 = ifftshift(&fftn(&fftshift(source)));
    let ft_u2 = c * fft_f1 * dx1.powi(2);
    return ft_u2.into_dimensionality().unwrap();
}
