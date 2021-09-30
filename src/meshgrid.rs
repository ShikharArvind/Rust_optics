use ndarray::prelude::*;

pub fn meshgrid<T: std::clone::Clone>(x: &Array1<T>, y: &Array1<T>) -> (Array2<T>, Array2<T>) {
    // For A - shape (len(y) x len(x)) , B - shape(len(x) x len(y))  (rows x columns)
    // Supports only 1D -> 2D, might implement 3D too later TODO
    // Refer MATLAB documentation on meshgrid
    let a = (*x).clone();
    let b = (*y).clone();
    (
        a.broadcast((b.len(), a.len())).unwrap().into_owned(),
        b.broadcast((a.len(), b.len()))
            .unwrap()
            .reversed_axes()
            .into_owned(),
    )
}
