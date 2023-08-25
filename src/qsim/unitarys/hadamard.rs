use crate::qsim::math::get_amount_bits;
use crate::qsim::math::multiply_matrix_vector;
use crate::qsim::math::tensor_matrix_matrix;
use num_traits::One;
use num_traits::Zero;
use std::ops::Mul;

use crate::qsim::math::Complex;
use crate::qsim::unitarys::Gate;

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
    fn apply(&self, state_vec: &mut Vec<f64>) {
        let total_qubits = get_amount_bits(&state_vec);
        assert!(self.apply_register_index < total_qubits, "The apply_register of the Hadamard gate: {} exceeds the amount of bits in the register {}", total_qubits, self.apply_register_index);

        let computed_hadamard: Vec<Vec<f64>> = self.dynamic_hadamard(total_qubits);
        let computed_state_vec = multiply_matrix_vector(&computed_hadamard, &state_vec);

        for i in 0..state_vec.len() {
            state_vec[i] = computed_state_vec[i];
        }
    }
}

impl Hadamard<Complex<f64>> {
    pub fn new(apply_register_index: usize) -> Self {
        let value = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let matrix = vec![
            vec![value, value],
            vec![value, Complex::new(-1.0 / 2.0f64.sqrt(), 0.0)],
        ];
        Self {
            apply_register_index,
            matrix,
        }
    }

    fn dynamic_hadamard(&self, total_qubits: usize) -> Vec<Vec<Complex<f64>>> {
        let identity_matrix = vec![
            vec![Complex::<f64>::one(), Complex::<f64>::zero()],
            vec![Complex::<f64>::zero(), Complex::<f64>::one()],
        ];
        let mut register_identity_matrix_vec = vec![identity_matrix; total_qubits];
        register_identity_matrix_vec[self.apply_register_index] = self.matrix.clone();

        let mut computed_hadamard: Vec<Vec<Complex<f64>>> = vec![vec![Complex::<f64>::one(); 1]];
        for matrix in &register_identity_matrix_vec {
            let rs = tensor_matrix_matrix(&computed_hadamard, matrix);
            computed_hadamard = rs;
        }
        computed_hadamard
    }
}

impl Gate<Complex<f64>> for Hadamard<Complex<f64>> {
    fn apply(&self, state_vec: &mut Vec<Complex<f64>>) {
        let total_qubits = get_amount_bits(&state_vec);
        assert!(self.apply_register_index < total_qubits, "The apply_register of the Hadamard gate: {} exceeds the amount of bits in the register {}", total_qubits, self.apply_register_index);

        let computed_hadamard: Vec<Vec<Complex<f64>>> = self.dynamic_hadamard(total_qubits);
        let computed_state_vec = multiply_matrix_vector(&computed_hadamard, &state_vec);
        for i in 0..state_vec.len() {
            state_vec[i] = computed_state_vec[i];
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_hadamard_f64() {
        let mut ket_0 = vec![1.0, 0.0];

        let h = Hadamard::<f64>::new(0);

        h.apply(&mut ket_0);
        assert_eq!(ket_0, vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()]);
    }

    #[test]
    fn test_hadamard_complexf64() {
        let o = Complex::one();
        let z = Complex::zero();
        let mut ket_0 = vec![o, z];

        let h = Hadamard::<Complex<f64>>::new(0);

        let result = h.apply(&mut ket_0);
        let a = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);

        assert_eq!(ket_0, vec![a, a]);
    }
}
