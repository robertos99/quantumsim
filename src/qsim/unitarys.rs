use crate::qsim::math::get_amount_bits;
use crate::qsim::math::multiply_matrix_vector;
use crate::qsim::math::tensor_matrix_matrix;
use std::ops::Mul;

pub trait Gate<T>
where
    T: Mul<Output = T> + Copy,
{
    fn apply(&self, state_vec: Vec<T>) -> Vec<T>;
}

pub struct Hadamard<T>
where
    T: Mul<Output = T> + Copy,
{
    apply_register_index: usize,
    matrix: Vec<Vec<T>>,
}

impl Hadamard<f64> {
    pub fn new(apply_register_index: usize) -> Self {
        let value = 1.0 / 2.0f64.sqrt();
        let matrix = vec![vec![value, value], vec![value, -value]];
        Self {
            apply_register_index,
            matrix,
        }
    }

    fn dynamic_hadamard(&self, total_qubits: usize) -> Vec<Vec<f64>> {
        let identity_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let mut register_identity_matrix_vec = vec![identity_matrix; total_qubits];
        register_identity_matrix_vec[self.apply_register_index] = self.matrix.clone();

        let mut computed_hadamard: Vec<Vec<f64>> = vec![vec![1f64; 1]];
        for matrix in &register_identity_matrix_vec {
            let rs = tensor_matrix_matrix(&computed_hadamard, matrix);
            computed_hadamard = rs;
        }
        computed_hadamard
    }
}

impl Gate<f64> for Hadamard<f64> {
    fn apply(&self, state_vec: Vec<f64>) -> Vec<f64> {
        let total_qubits = get_amount_bits(&state_vec);
        assert!(self.apply_register_index < total_qubits, "The apply_register of the Hadamard gate: {} exceeds the amount of bits in the register {}", total_qubits, self.apply_register_index);

        let computed_hadamard: Vec<Vec<f64>> = self.dynamic_hadamard(total_qubits);
        let computed_state_vec = multiply_matrix_vector(&computed_hadamard, &state_vec);
        computed_state_vec
    }
}

pub struct CNot {
    control: usize,
    target: usize,
}

impl CNot {
    pub fn new(control: usize, target: usize) -> Self {
        Self { control, target }
    }

    fn dynamic_cnot(&self, total_qubits: usize) -> Vec<Vec<f64>> {
        let control = self.control;
        let target = self.target;

        let cnot = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let id = identity(2);

        let mut result = vec![vec![1.0]];

        let mut i = 0;
        while i < total_qubits {
            if i == control {
                if i + 1 == target {
                    // Adjusted for 0-indexing
                    result = tensor_matrix_matrix(&result, &cnot);
                    i += 2;
                    continue;
                } else {
                    result = tensor_matrix_matrix(&result, &id);
                }
            } else if i == target {
                result = tensor_matrix_matrix(&result, &id);
            } else {
                result = tensor_matrix_matrix(&result, &id);
            }
            i += 1;
        }
        result
    }
}

use crate::qsim::math::identity;

impl Gate<f64> for CNot {
    fn apply(&self, state_vec: Vec<f64>) -> Vec<f64> {
        let cnot = self.dynamic_cnot(get_amount_bits(&state_vec));
        multiply_matrix_vector(&cnot, &state_vec)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_hadamard() {
        let ket_0 = vec![1.0, 0.0];

        let h = Hadamard::new(0);

        let result = h.apply(ket_0);
        assert_eq!(result, vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()]);
    }

    #[test]
    fn test_cnot() {
        // |00> and hadamard the first bit -> |+0>
        // one step before epr pair
        let pre_epr = vec![1.0 / 2.0f64.sqrt(), 0.0, 1.0 / 2.0f64.sqrt(), 0.0];

        let h = CNot::new(0, 1);

        // 1/sqrt(2) |00> + 1/sqrt(2) |11>
        // Bell state / epr-pair
        let result = h.apply(pre_epr);

        print!("{:?}", result);
        assert_eq!(
            result,
            vec![1.0 / 2.0f64.sqrt(), 0.0, 0.0, 1.0 / 2.0f64.sqrt()]
        );
    }
}
