use ndarray::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
pub fn write_to_file<T: std::fmt::Display>(file_name: &String, input: &Array2<T>) {
    let mut f = BufWriter::new(File::create(file_name).expect("Unable to create file"));

    write!(f, "[").expect("Writing failed");
    for i in 0..(*input).len_of(Axis(0)) {
        // write the ith row with out comma
        write!(f, "[").expect("Writing failed");
        for iter in (*input).index_axis(Axis(1), i).indexed_iter() {
            if iter.0 == (*input).len_of(Axis(1)) - 1 {
                // write the last element of the row with out comma
                write!(f, "{}", *(iter.1)).expect("Writing failed");
            } else {
                // write the all elements of the row except last separated by comma
                write!(f, "{},", *(iter.1)).expect("Writing failed");
            }
        }
        if i == (*input).len_of(Axis(0)) - 1 {
            // close the ith row with ] and no line break
            write!(f, "]").expect("Writing failed");
        } else {
            // close the last row with ] and line break (this is wrong TODO)
            writeln!(f, "]").expect("Writing failed");
        }
    }
    write!(f, "]").expect("Writing failed");
    f.flush().unwrap();
}
