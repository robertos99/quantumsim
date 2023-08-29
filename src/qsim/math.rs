use num_traits::Float;
use num_traits::One;
use num_traits::Zero;
use std::ops::Add;
use std::ops::Mul;

pub trait Magnitude {
    type Output;

    fn magnitude(&self) -> Self::Output;
}

impl Magnitude for f64 {
    type Output = f64;

    fn magnitude(&self) -> Self::Output {
        *self
    }
}

impl Magnitude for f32 {
    type Output = f32;

    fn magnitude(&self) -> Self::Output {
        *self
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Complex<T>
where
    T: Float,
{
    real: T,
    imagin: T,
}

impl<T: Float> Complex<T> {
    pub fn new(real: T, imagin: T) -> Self {
        Self { real, imagin }
    }
}

impl Magnitude for Complex<f64> {
    type Output = f64;
    fn magnitude(&self) -> Self::Output {
        (self.real + self.imagin).sqrt()
    }
}

impl Add for Complex<f64> {
    type Output = Self;

    fn add(self, rhs: Complex<f64>) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            imagin: self.imagin + rhs.imagin,
        }
    }
}
impl Zero for Complex<f64> {
    fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imagin == 0.0
    }

    fn set_zero(&mut self) {
        *self = Self::zero();
    }

    fn zero() -> Self {
        Self {
            real: 0.0,
            imagin: 0.0,
        }
    }
}

impl Mul for Complex<f64> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.real;
        let b = self.imagin;
        let c = rhs.real;
        let d = rhs.imagin;

        Self {
            real: a * c - b * d,
            imagin: a * d + b * c,
        }
    }
}

impl One for Complex<f64> {
    fn one() -> Self {
        Complex {
            real: 1.0,
            imagin: 0.0,
        }
    }
}

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
    T: Mul<Output = T> + Add + Copy + Zero,
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
            result[i] = result[i] + (matrix[i][j] * vector[j]);
        }
    }

    result
}

pub fn identity<T: One + Zero + Clone>(dim: usize) -> Vec<Vec<T>> {
    let mut matrix = vec![vec![T::zero(); dim]; dim];
    for i in 0..dim {
        matrix[i][i] = T::one();
    }
    return matrix;
}

#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn test_complex_mul() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(3.0, 4.0);

        let result = a.mul(b);

        let expected = Complex::new(-7.0, 24.0);

        assert_eq!(expected, result);
    }

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
