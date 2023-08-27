use crate::qsim::{
    math::{get_amount_bits, multiply_matrix_vector},
    unitarys::Gate,
};

use num_traits::{One, Zero};

pub struct CCNot {
    control_1: usize,
    control_2: usize,
    target: usize,
}

impl CCNot {
    pub fn new(control_1: usize, control_2: usize, target: usize) -> Self {
        Self {
            control_1,
            control_2,
            target,
        }
    }

    fn dynamic_ccnot<T: Zero + One + Clone>(&self, register_size: usize) -> Vec<Vec<T>> {
        let control_1 = self.control_1;
        let control_2 = self.control_2;
        let target = self.target;
        assert!(
            control_1 < register_size && control_2 < register_size && target < register_size,
            "Control_1: {}, Control_2: {} or Target: {} out of register bounds.",
            control_1,
            control_2,
            target
        );
        let dim = 1 << register_size;

        let mut ccnot_matrix = vec![vec![T::zero(); dim]; dim];

        for i in 0..dim {
            let c1_set = 1 << register_size - 1 - control_1;
            let c2_set = 1 << register_size - 1 - control_2;
            // check if control1 and control2 bit are both 1 at their respective index
            // 110: true for control_1 = 0 and control_2 = 1.
            // we are checking from the left side based on Dirac notation |110> (not the usize left most bit)
            if i & c1_set != 0 && i & c2_set != 0 {
                // flippin' the "bit", by swapping the amplitude of the respective dimension
                let flipped_bit = 1 << register_size - 1 - target;
                ccnot_matrix[i][i ^ flipped_bit] = T::one();
            } else {
                ccnot_matrix[i][i] = T::one();
            }
        }

        ccnot_matrix
    }
}

impl Gate<f64> for CCNot {
    fn apply(&self, state_vec: &mut Vec<f64>) {
        let ccnot_matrix = self.dynamic_ccnot::<f64>(get_amount_bits(state_vec));
        let computed_state_vec = multiply_matrix_vector(&ccnot_matrix, state_vec);
        for i in 0..state_vec.len() {
            state_vec[i] = computed_state_vec[i];
        }
    }
}

#[cfg(test)]
mod test {
    use std::vec;

    use super::*;

    #[test]
    fn test_dynamic_ccnot_f64_0_1_2() {
        let control_1 = 0;
        let control_2 = 1;
        let target = 2;

        let ccnot = CCNot::new(control_1, control_2, target);
        let result = ccnot.dynamic_ccnot(3);

        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_dynamic_ccnot_f64_1_2_0() {
        let control_1 = 1;
        let control_2 = 2;
        let target = 0;

        let ccnot = CCNot::new(control_1, control_2, target);
        let result = ccnot.dynamic_ccnot(3);

        let expected = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];

        assert_eq!(expected, result);
    }

    #[test]
    fn test_apply_f64_0_1_2() {
        let control_1 = 0;
        let control_2 = 1;
        let target = 2;

        let ccnot = CCNot::new(control_1, control_2, target);

        let mut state_vec = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        ccnot.apply(&mut state_vec);

        let expected = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.7];

        assert_eq!(expected, state_vec);
    }

    #[test]
    fn test_apply_f64_1_2_0() {
        let control_1 = 1;
        let control_2 = 2;
        let target = 0;

        let ccnot = CCNot::new(control_1, control_2, target);

        let mut state_vec = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        ccnot.apply(&mut state_vec);

        let expected = vec![0.1, 0.2, 0.3, 0.8, 0.5, 0.6, 0.7, 0.4];

        assert_eq!(expected, state_vec);
    }
}
