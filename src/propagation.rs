use super::fftshift;
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

    //FT(FTSHIFT(u1))-> FT_along_axis_1(FT_along_axis_0(FTSHIFT(u1)))
    //There should be a cleaner way to implement (TODO)
    let shift_u1 = fftshift::fftshiftgen(&(u1_.into_dyn()));
    let mut fftx_u1 = ArrayD::<Complex64>::zeros(shift_u1.shape());
    let mut handler = FftHandler::new(shift_u1.len_of(Axis(0)));
    ndfft(&shift_u1, &mut fftx_u1, &mut handler, 0);
    handler = FftHandler::new(shift_u1.len_of(Axis(1)));
    let mut ffty_u1 = ArrayD::<Complex64>::zeros(fftx_u1.shape());
    ndfft(&fftx_u1, &mut ffty_u1, &mut handler, 1);
    let fft_f1 = ffty_u1;

    // FT(u2) =  FT(FTSHIFT(u1)) * FTSHIFT(H)
    let ft_u2 = fft_f1 * fftshift::fftshiftgen(&(H.into_dyn()));

    // u2 = IFFTSHIFT(FT_along_axis_1(FT_along_axis_0(ft_u1)))
    // There should be a cleaner way to implement (TODO)
    handler = FftHandler::new(ft_u2.len_of(Axis(0)));
    let mut ifftx_u2 = ArrayD::<Complex64>::zeros(ft_u2.shape());
    ndifft(&ft_u2, &mut ifftx_u2, &mut handler, 0);
    handler = FftHandler::new(ifftx_u2.len_of(Axis(1)));
    let mut iffty_u2 = ArrayD::<Complex64>::zeros(ifftx_u2.shape());
    ndifft(&mut ifftx_u2, &mut iffty_u2, &mut handler, 1);
    let ifft_u2 = iffty_u2;

    return fftshift::ifftshiftgen(&ifft_u2)
        .into_dimensionality()
        .unwrap();
}
