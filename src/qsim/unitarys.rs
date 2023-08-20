use crate::qsim::math::get_amount_bits;
use crate::qsim::math::identity;
use crate::qsim::math::multiply_matrix_vector;
use crate::qsim::math::tensor_matrix_matrix;
use num_traits::One;
use num_traits::Zero;
use std::ops::Mul;

use super::math::Complex;
pub trait Gate<T>
where
    T: Mul<Output = T> + Copy,
{
    fn apply(&self, state_vec: &Vec<T>) -> Vec<T>;
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
    fn apply(&self, state_vec: &Vec<f64>) -> Vec<f64> {
        let total_qubits = get_amount_bits(&state_vec);
        assert!(self.apply_register_index < total_qubits, "The apply_register of the Hadamard gate: {} exceeds the amount of bits in the register {}", total_qubits, self.apply_register_index);

        let computed_hadamard: Vec<Vec<f64>> = self.dynamic_hadamard(total_qubits);
        let computed_state_vec = multiply_matrix_vector(&computed_hadamard, &state_vec);
        computed_state_vec
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
    fn apply(&self, state_vec: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        let total_qubits = get_amount_bits(&state_vec);
        assert!(self.apply_register_index < total_qubits, "The apply_register of the Hadamard gate: {} exceeds the amount of bits in the register {}", total_qubits, self.apply_register_index);

        let computed_hadamard: Vec<Vec<Complex<f64>>> = self.dynamic_hadamard(total_qubits);
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

    ///
    /// function to create a dynamic cnot matrix.
    /// This is required to react to different sized registers. This also depends on which bit is target and control.
    /// !!! target and control are 0 indexed !!!
    /// !!! The 0 index is the left outer bit of the register. This means the |011> the 0 is at index 0 !!!
    fn dynamic_cnot<T: Zero + One + Clone>(&self, register_size: usize) -> Vec<Vec<T>> {
        let control = self.control;
        let target = self.target;
        let dim = 1 << register_size; // 2^register_size
        let mut cnot_matrix = vec![vec![T::zero(); dim]; dim];

        for i in 0..dim {
            if (i & (1 << register_size - 1 - control)) != 0 {
                // if control bit is set, flip the target bit
                cnot_matrix[i][i ^ (1 << register_size - 1 - target)] = T::one();
            } else {
                // if control bit is not set, state remains unchanged
                cnot_matrix[i][i] = T::one();
            }
        }

        cnot_matrix
    }
}

impl Gate<f64> for CNot {
    fn apply(&self, state_vec: &Vec<f64>) -> Vec<f64> {
        let cnot = self.dynamic_cnot::<f64>(get_amount_bits(&state_vec));
        multiply_matrix_vector(&cnot, &state_vec)
    }
}

impl Gate<Complex<f64>> for CNot {
    fn apply(&self, state_vec: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
        let cnot = self.dynamic_cnot::<Complex<f64>>(get_amount_bits(&state_vec));
        multiply_matrix_vector(&cnot, &state_vec)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_hadamard_f64() {
        let ket_0 = vec![1.0, 0.0];

        let h = Hadamard::<f64>::new(0);

        let result = h.apply(&ket_0);
        assert_eq!(result, vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt()]);
    }

    #[test]
    fn test_cnot_f64() {
        // |00> and hadamard the first bit -> |+0>
        // one step before epr pair
        let pre_epr = vec![1.0 / 2.0f64.sqrt(), 0.0, 1.0 / 2.0f64.sqrt(), 0.0];

        let h = CNot::new(0, 1);

        // 1/sqrt(2) |00> + 1/sqrt(2) |11>
        // Bell state / epr-pair
        let result = h.apply(&pre_epr);

        print!("{:?}", result);
        assert_eq!(
            result,
            vec![1.0 / 2.0f64.sqrt(), 0.0, 0.0, 1.0 / 2.0f64.sqrt()]
        );
    }

    #[test]
    fn test_hadamard_complexf64() {
        let o = Complex::one();
        let z = Complex::zero();
        let ket_0 = vec![o, z];

        let h = Hadamard::<Complex<f64>>::new(0);

        let result = h.apply(&ket_0);
        let a = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);

        assert_eq!(result, vec![a, a]);
    }

    #[test]
    fn test_cnot_complexf64() {
        // |00> and hadamard the first bit -> |+0>
        // one step before epr pair
        let a = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let z = Complex::zero();
        let pre_epr = vec![a, z, a, z];

        let h = CNot::new(0, 1);

        // 1/sqrt(2) |00> + 1/sqrt(2) |11>
        // Bell state / epr-pair
        let result = h.apply(&pre_epr);

        print!("{:?}", result);
        assert_eq!(result, vec![a, z, z, a]);
    }

    #[test]
    fn test_dynamic_cnot_f64_3bit_0_1() {
        // control: first bit (index 0)
        // target: second bit (index 1)
        let cnot = CNot::new(0, 1);
        let result = cnot.dynamic_cnot::<f64>(3);
        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ];
        assert_eq!(expected, result);
    }

    #[test]
    fn test_dynamic_cnot_f64_3bit_0_2() {
        // control: first bit (index 0)
        // target: third bit (index 2)
        let cnot = CNot::new(0, 2);
        let result = cnot.dynamic_cnot::<f64>(3);
        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ];
        assert_eq!(expected, result);
    }

    #[test]
    fn test_dynamic_cnot_f64_3bit_2_0() {
        // control: first bit (index 0)
        // target: third bit (index 2)
        let cnot = CNot::new(2, 0);
        let result = cnot.dynamic_cnot::<f64>(3);
        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        assert_eq!(expected, result);
    }

    #[test]
    fn test_dynamic_cnot_f64_2bit_0_1() {
        // control: first bit (index 0)
        // target: second bit (index 1)
        let cnot = CNot::new(0, 1);
        let result = cnot.dynamic_cnot::<f64>(2);
        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        assert_eq!(expected, result);
    }
}
