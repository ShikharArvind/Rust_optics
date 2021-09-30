use super::fft::{fftn, fftshift, ifftn, ifftshift};
use super::meshgrid;
use ndarray::prelude::*;
use ndrustfft::*;
use num_complex::Complex64;
use std::f64::consts::PI;

pub fn propTF(u1: &Array2<Complex64>, l: &f64, lam: &f64, z: &f64) -> Array2<Complex64> {
    let u1_ = (*u1).clone();
    let l_ = (*l).clone();
    let lam_ = (*lam).clone();
    let z_ = (*z).clone();
    //let n: f64 = u1_.len_of(Axis(0)) as f64;
    let m: f64 = u1_.len_of(Axis(1)) as f64;
    let dx: f64 = l_ / m;
    let fx_range = Array1::<f64>::range(-1.0 / (2.0 * dx), 1.0 / (2.0 * dx), 1.0 / l_);
    let k = (2.0 * PI) / lam_;
    let (mesh_fx, mesh_fy) = meshgrid::meshgrid(&fx_range, &fx_range);
    //H - transfer function
    let H = (Complex64::i() * k * z_).exp()
        * (-1.0
            * Complex64::i()
            * PI
            * lam_
            * z_
            * (mesh_fx.mapv(|x| Complex64::new(x.powi(2), 0.0))
                + mesh_fy.mapv(|y| Complex64::new(y.powi(2), 0.0))))
        .mapv(|a| a.exp());

    let fft_f1 = fftn(&fftshift(&(u1_.into_dyn())));
    let ft_u2 = fft_f1 * fftshift(&(H.into_dyn()));
    return ifftshift(&ifftn(&ft_u2)).into_dimensionality().unwrap();
}
