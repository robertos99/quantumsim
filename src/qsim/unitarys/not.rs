use num_traits::{One, Zero};

use crate::qsim::math::{get_amount_bits, multiply_matrix_vector};

use super::Gate;

pub struct Not {
    target: usize,
}

impl Not {
    pub fn new(target: usize) -> Self {
        Self { target }
    }

    fn dynamic_not<T: Zero + One + Clone>(&self, register_size: usize) -> Vec<Vec<T>> {
        let target_bit = self.target;
        let dim = 1 << register_size;
        let mut not_matrix = vec![vec![T::zero(); dim]; dim];

        for i in 0..dim {
            let flipped = i ^ (1 << register_size - 1 - target_bit);
            not_matrix[i][flipped] = T::one();
        }

        not_matrix
    }
}

impl Gate<f64> for Not {
    fn apply(&self, state_vec: &mut Vec<f64>) {
        let not_matrix = Self::dynamic_not::<f64>(&self, get_amount_bits(state_vec));
        let result = multiply_matrix_vector(&not_matrix, state_vec);

        for i in 0..result.len() {
            state_vec[i] = result[i];
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_dynamic_not_f64_0() {
        let not = Not::new(0);

        let result = not.dynamic_not::<f64>(2);

        let expected = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        assert_eq!(result, expected);
    }
    #[test]
    fn test_apply_f64_0() {
        let not = Not::new(0);

        let mut state_vec = vec![0.1, 0.2, 0.3, 0.4];

        not.apply(&mut state_vec);
        let expected = vec![0.3, 0.4, 0.1, 0.2];
        assert_eq!(state_vec, expected);
    }
}
