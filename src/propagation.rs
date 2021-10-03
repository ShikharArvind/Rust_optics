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
