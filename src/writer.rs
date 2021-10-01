use ndarray::prelude::*;
use std::fs::File;
use std::io::Write;

pub fn writer2d<T: std::fmt::Display>(file_name: &String, input: &Array2<T>) {
    let mut f = File::create(file_name).expect("Unable to create file");

    write!(f, "[").expect("Writing failed");
    for i in 0..(*input).len_of(Axis(0)) {
        write!(f, "[").expect("Writing failed");
        for iter in (*input).index_axis(Axis(1), i).indexed_iter() {
            if iter.0 == (*input).len_of(Axis(1)) - 1 {
                write!(f, "{}", *(iter.1)).expect("Writing failed");
            } else {
                write!(f, "{},", *(iter.1)).expect("Writing failed");
            }
        }
        if i == (*input).len_of(Axis(0)) - 1 {
            write!(f, "]").expect("Writing failed");
        } else {
            writeln!(f, "]").expect("Writing failed");
        }
    }
    write!(f, "]").expect("Writing failed");
}
