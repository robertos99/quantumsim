use crate::qsim::math::get_amount_bits;
use crate::qsim::math::multiply_matrix_vector;
use num_traits::One;
use num_traits::Zero;


use crate::qsim::math::Complex;
use crate::qsim::unitarys::Gate;

pub struct CNot {
    control: usize,
    target: usize,
}

impl CNot {
    pub fn new(control: usize, target: usize) -> Self {
        Self { control, target }
    }


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


#[cfg(test)]
mod test {
    use super::*;

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

}