use crate::qsim::math::get_amount_bits;
use crate::qsim::math::identity;
use crate::qsim::math::multiply_matrix_vector;
use crate::qsim::math::tensor_matrix_matrix;
use num_traits::One;
use num_traits::Zero;
use std::ops::Mul;

use crate::qsim::math::Complex;
pub trait Gate<T>
where
    T: Mul<Output = T> + Copy,
{
    // TODO make this mutate the vector so that we dont waste memory
    fn apply(&self, state_vec: &mut Vec<T>);
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
    fn apply(&self, state_vec: &mut Vec<f64>)  {
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
    fn apply(&self, state_vec: &mut Vec<Complex<f64>>){
        let total_qubits = get_amount_bits(&state_vec);
        assert!(self.apply_register_index < total_qubits, "The apply_register of the Hadamard gate: {} exceeds the amount of bits in the register {}", total_qubits, self.apply_register_index);

        let computed_hadamard: Vec<Vec<Complex<f64>>> = self.dynamic_hadamard(total_qubits);
        let computed_state_vec = multiply_matrix_vector(&computed_hadamard, &state_vec);
        for i in 0..state_vec.len() {
            state_vec[i] = computed_state_vec[i];
        }
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

        let mut cnot_matrix = vec![vec![T::zero(); dim]; dim]; // create zeroed matrix which we fill in later

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
    fn apply(&self, state_vec: &mut Vec<f64>) {
        let cnot = self.dynamic_cnot::<f64>(get_amount_bits(&state_vec));
        let computed_state_vec =  multiply_matrix_vector(&cnot, &state_vec);
        for i in 0..state_vec.len() {
            state_vec[i] = computed_state_vec[i];
        }
    }
}

impl Gate<Complex<f64>> for CNot {
    fn apply(&self, state_vec: &mut Vec<Complex<f64>>){
        let bits_in_register = get_amount_bits(&state_vec);
        assert!(self.control < bits_in_register && self.target < bits_in_register, 
            "CNot control or target bit out of register bound. Register holds {} bits. Control was {}. Target was {}.", 
            bits_in_register, self.control, self.target);
        let cnot = self.dynamic_cnot::<Complex<f64>>(bits_in_register);
        let computed_state_vec = multiply_matrix_vector(&cnot, &state_vec);
        for i in 0..state_vec.len() {
            state_vec[i] = computed_state_vec[i];
        }
    }
}

struct CSwap {
    control: usize,
    target_1: usize,
    target_2: usize,
}

impl CSwap {
    pub fn new(control: usize, target_1: usize, target_2: usize) -> Self {
        let are_equal = control == target_1 || control == target_2 || target_1 == target_2;

        Self {
            control,
            target_1,
            target_2,
        }
    }

    fn dynamic_cswap<T: Zero + One + Clone>(&self, register_size: usize) -> Vec<Vec<T>> {
        let control = self.control;
        let target_1 = self.target_1;
        let target_2 = self.target_2;
        let dim = 1 << register_size; // 2^register_size

        let mut cswap_matrix = vec![vec![T::zero(); dim]; dim]; // create zeroed matrix which we fill in later

        for i in 0..dim {
            if (i & (1 << register_size - 1 - control)) != 0 {
                let tbit_1 = i & (1 << register_size - 1 - target_1);
                let tbit_2 = i & (1 << register_size - 1 - target_2);

                if (tbit_1 == 0 && tbit_2 != 0) || (tbit_2 == 0 && tbit_1 != 0) {
                    let swap_mask = i ^ (1 << register_size - 1 - target_1);
                    let swap_mask = swap_mask ^ (1 << register_size - 1 - self.target_2);
                    cswap_matrix[i][swap_mask] = T::one();
                } else {
                    // if control bit is not set, state remains unchanged
                    cswap_matrix[i][i] = T::one();
                }
            } else {
                // if control bit is not set, state remains unchanged
                cswap_matrix[i][i] = T::one();
            }
        }

        cswap_matrix
    }
}

impl Gate<f64> for CSwap {
    fn apply(&self, state_vec: &mut Vec<f64>) {
        let bits_in_register = get_amount_bits(&state_vec);
        assert!(self.control < bits_in_register && self.target_1 < bits_in_register && self.target_2 < bits_in_register, 
            "CSwap control or target_1 or target_2 bit out of register bound. Register holds {} bits. Control was {}. Target_1 was {}. Target_2 was {}.", 
            bits_in_register, self.control, self.target_1, self.target_2);
        let cswap = self.dynamic_cswap::<f64>(get_amount_bits(&state_vec));
        let computed_state_vec = multiply_matrix_vector(&cswap, &state_vec);
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
    fn test_cnot_f64() {
        // |00> and hadamard the first bit -> |+0>
        // one step before epr pair
        let mut pre_epr = vec![1.0 / 2.0f64.sqrt(), 0.0, 1.0 / 2.0f64.sqrt(), 0.0];

        let h = CNot::new(0, 1);

        // 1/sqrt(2) |00> + 1/sqrt(2) |11>
        // Bell state / epr-pair
        h.apply(&mut pre_epr);

        print!("{:?}", pre_epr);
        assert_eq!(
            pre_epr,
            vec![1.0 / 2.0f64.sqrt(), 0.0, 0.0, 1.0 / 2.0f64.sqrt()]
        );
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

    #[test]
    fn test_cnot_complexf64() {
        // |00> and hadamard the first bit -> |+0>
        // one step before epr pair
        let a = Complex::new(1.0 / 2.0f64.sqrt(), 0.0);
        let z = Complex::zero();
        let mut pre_epr = vec![a, z, a, z];

        let h = CNot::new(0, 1);

        // 1/sqrt(2) |00> + 1/sqrt(2) |11>
        // Bell state / epr-pair
        h.apply(&mut pre_epr);

        assert_eq!(pre_epr, vec![a, z, z, a]);
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

    #[test]
    fn test_dynamic_cswap_f64_adjacent_in_order() {
        let cswap = CSwap::new(0, 1, 2);
        let result = cswap.dynamic_cswap::<f64>(3);
        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(expected, result);
    }

    #[test]
    fn test_dynamic_cswap_f64_adjacent_not_in_order() {
        let cswap = CSwap::new(0, 2, 1);
        let result = cswap.dynamic_cswap::<f64>(3);
        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(expected, result);
    }

    #[test]
    fn test_dynamic_cswap_f64_non_adjacent_in_order() {
        let cswap = CSwap::new(0, 1, 3);
        let result = cswap.dynamic_cswap::<f64>(4);
        //1001 -> 1100; 1011 -> 1110; 1100 -> 1001; 1110 -> 1011
        let expected = vec![
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ];
        assert_eq!(expected, result);
    }

    #[test]
    fn test_dynamic_cswap_f64_non_adjacent_out_of_order() {
        let cswap = CSwap::new(0, 3, 1); // Note the change in order here for target_1 and target_2
        let result = cswap.dynamic_cswap::<f64>(4);
        let expected = vec![
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ];
        assert_eq!(expected, result);
    }
}
