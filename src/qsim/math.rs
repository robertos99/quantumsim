use num_traits::Zero;

use std::ops::AddAssign;
use std::ops::Mul;

pub fn tensor_vector_vector<T>(u: &[T], v: &[T]) -> Vec<T>
where
    T: Mul<Output = T> + Copy,
{
    let mut result = Vec::with_capacity(u.len() * v.len());

    for &a in u.iter() {
        for &b in v.iter() {
            result.push(a * b);
        }
    }

    result
}

pub fn tensor_matrix_matrix<T>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Mul<Output = T> + Copy,
{
    let mut result = Vec::new();

    for row_a in a.iter() {
        for row_b in b.iter() {
            let mut new_row = Vec::new();
            for &val_a in row_a.iter() {
                for &val_b in row_b.iter() {
                    new_row.push(val_a * val_b);
                }
            }
            result.push(new_row);
        }
    }

    result
}

pub fn get_amount_bits<T>(bits: &[T]) -> usize {
    let mut big_n = bits.len();
    let mut n = 0;
    while big_n > 1 {
        big_n = big_n >> 1;
        n += 1;
    }
    n
}

pub fn multiply_matrix_vector<T>(matrix: &Vec<Vec<T>>, vector: &Vec<T>) -> Vec<T>
where
    T: Mul<Output = T> + AddAssign + Copy + Zero,
{
    let rows = matrix.len();
    let cols = matrix[0].len();

    assert_eq!(
        cols,
        vector.len(),
        "Number of columns in the matrix must equal the number of rows in the vector."
    );

    let mut result = vec![T::zero(); rows];

    for i in 0..rows {
        for j in 0..cols {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    result
}

pub fn identity(dim: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        matrix[i][i] = 1.0;
    }
    return matrix;
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_get_amount_bits() {
        let cases = vec![
            (vec![0; 2], 1),  // 2^1 = 2
            (vec![0; 4], 2),  // 2^2 = 4
            (vec![0; 8], 3),  // 2^3 = 8
            (vec![0; 16], 4), // 2^4 = 16
            (vec![0; 32], 5), // 2^5 = 32
            (vec![0; 64], 6), // 2^6 = 64
        ];

        for (input, expected) in cases.iter() {
            assert_eq!(get_amount_bits(&input), *expected);
        }
    }
}
