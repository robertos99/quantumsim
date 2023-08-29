use std::vec;

use crate::qsim::math::get_amount_bits;
use crate::qsim::unitarys::Gate;
use num_traits::One;
use num_traits::Zero;
use std::ops::Mul;

use super::math::Magnitude;

pub struct QuantumRegister<T>
where
    T: Mul<Output = T> + Copy + Magnitude + Zero + One,
{
    state_vector: Vec<T>,
    gates: Vec<Box<dyn Gate<T>>>,
}

impl QuantumRegister<f64>
//where

//T: Mul<Output = T> + Copy + Magnitude<Output = f64> + Zero + One,
{
    pub fn new(amount_qubits: usize) -> Self {
        // This is simply to prevent me from creating registers that become too large.
        // Can be adjusted as desired.
        assert!(
            amount_qubits < 9,
            "Bit index {} too large. Maximum allowed is 8.",
            amount_qubits
        );

        let mut init = vec![f64::zero(); 2_usize.pow(amount_qubits as u32)];
        init[0] = f64::one();
        QuantumRegister {
            state_vector: init,
            gates: vec![],
        }
    }

    pub fn add_gate(&mut self, gate: Box<dyn Gate<f64>>) {
        self.gates.push(gate);
    }

    pub fn run(&mut self) {
        for gate in &self.gates {
            gate.apply(&mut self.state_vector);
        }
    }

    pub fn measure_no_collapse(&mut self, i: usize) -> u8 {
        let amount_bits_in_register = get_amount_bits(&self.state_vector);
        // Register is 0 indexed.
        assert!(
            i < amount_bits_in_register,
            "i: {} is too large. The register holds {} Bits and is indexed 0.",
            i,
            amount_bits_in_register
        );

        let mut prob_1 = 0_f64;
        for (index, value) in self.state_vector.iter().enumerate() {
            // Bitshifting to check if the index of current amplitude corresponds with a 1 bit for the i'th Qubit.
            // i = 0 for the left most bit in the register. Example: if the register holds |10> then i = 0 points to the Qubit that is in state 1.
            // amplitude-index(5)                   = 0000 0[1]01
            // mask for amount_qbits(3) and i(0)    = 0000 0[1]00
            // result                               = 0000 0[1]00 != 0 -> bit is 1
            if index & 1 << amount_bits_in_register - 1 - i != 0 {
                // sum of square of magnitude
                // we only need to save the prob_1 because prob_0 is simply 1 - prob_1
                prob_1 += value.magnitude().powi(2);
            }
        }

        let rand_num: f64 = rand::random();
        if rand_num < prob_1 {
            1
        } else {
            0
        }
    }

    pub fn measure_with_collapse(&mut self, i: usize) -> u8 {
        let amount_bits_in_register = get_amount_bits(&self.state_vector);
        // Register is 0 indexed.
        assert!(
            i < amount_bits_in_register,
            "i: {} is too large. The register holds {} Bits and is indexed 0.",
            i,
            amount_bits_in_register
        );

        let mut prob_1 = 0_f64;
        for (index, value) in self.state_vector.iter().enumerate() {
            // Bitshifting to check if the index of current amplitude corresponds with a 1 bit for the i'th Qubit.
            // i = 0 for the left most bit in the register. Example: if the register holds |10> then i = 0 points to the Qubit that is in state 1.
            // index(4)                     = 0000 0101
            // mask amount_qbits(4) - i(0)  = 0000 0100
            if index & 1 << amount_bits_in_register - 1 - i != 0 {
                // sum of square of magnitude
                // we only need to save the prob_1 because prob_0 is simply 1 - prob_1
                prob_1 += value.magnitude().powi(2);
            }
        }

        let rand_num: f64 = rand::random();
        let mut collapse = 0u8;
        if rand_num < prob_1 {
            collapse = 1;
        } else {
            collapse = 0;
        }

        for (index, value) in self.state_vector.iter_mut().enumerate() {
            if index & 1 << amount_bits_in_register - 1 - i != 0 {
                if collapse == 1 {
                    *value = *value / prob_1.sqrt();
                } else {
                    *value = 0.0;
                }
            } else {
                if collapse == 0 {
                    *value = *value / (1.0 - prob_1.sqrt());
                } else {
                    *value = 0.0;
                }
            }
        }
        collapse
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::qsim::{unitarys::cnot::CNot, unitarys::hadamard::Hadamard};

    #[test]
    fn test_new() {
        let n = 3; // for example
        let qr = QuantumRegister::<f64>::new(n);
        let expected_length = 2_u8.pow(n as u32) as usize;

        assert_eq!(qr.state_vector.len(), expected_length);
        assert_eq!(qr.state_vector[0], 1.0);

        for i in 1..expected_length {
            assert_eq!(qr.state_vector[i], 0.0);
        }
    }

    #[test]
    fn test_measure_valid_output() {
        let mut reg = QuantumRegister {
            state_vector: vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0, 0.0],
            gates: vec![],
        };

        // Check the measure function returns a valid value
        let result = reg.measure_no_collapse(0);
        assert!(result == 0 || result == 1);
    }

    #[test]
    #[should_panic(expected = "i: 3 is too large. The register holds 2 Bits and is indexed 0.")]
    fn test_measure_out_of_bounds_assertion() {
        let mut reg = QuantumRegister {
            state_vector: vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0, 0.0],
            gates: vec![],
        };

        // This should panic and thus pass the test. register holds 2 bits
        reg.measure_no_collapse(3);
    }

    #[test]
    fn test_measure_expected_probabilities() {
        // equivilant to |+0>
        let mut reg = QuantumRegister {
            state_vector: vec![1.0 / 2.0f64.sqrt(), 0.0, 1.0 / 2.0f64.sqrt(), 0.0],
            gates: vec![],
        };

        // Check with multiple runs to see if we roughly get expected probabilities
        // This is a probabilistic test. It might fail occasionally, but over a large number of runs,
        // the results should converge to the expected values.
        let num_runs = 10000;
        let mut ones_count = 0u32;
        for _ in 0..num_runs {
            ones_count += reg.measure_no_collapse(0) as u32;
        }
        let approx_prob_1 = ones_count as f64 / num_runs as f64;
        assert!((approx_prob_1 - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
    }

    #[test]
    fn test_bellstate_expected_probabilities_f64() {
        // This test doesnt account for collapse.
        // It only verifys that the states 0 and 1 are equally like for both bits.
        // There is a test that verifys that collapse behaves correctly.
        let mut reg = QuantumRegister::<f64>::new(2);
        let hadamard = Hadamard::<f64>::new(0);
        let cnot = CNot::new(0, 1);

        reg.add_gate(Box::new(hadamard));
        reg.add_gate(Box::new(cnot));
        reg.run();
        // Check with multiple runs to see if we roughly get expected probabilities
        // This is a probabilistic test. It might fail occasionally, but over a large number of runs,
        // the results should converge to the expected values.
        let num_runs = 10000;
        let mut ones_count_first_bit = 0u32;
        let mut ones_count_second_bit = 0u32;
        for _ in 0..num_runs {
            let first_bit = reg.measure_no_collapse(0) as u32;
            let second_bit = reg.measure_no_collapse(1) as u32;

            ones_count_first_bit += first_bit;
            ones_count_second_bit += second_bit;
        }
        let approx_prob_1_first_bit = ones_count_first_bit as f64 / num_runs as f64;
        let approx_prob_1_second_bit = ones_count_second_bit as f64 / num_runs as f64;
        assert!((approx_prob_1_first_bit - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
        assert!((approx_prob_1_second_bit - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
    }

    #[test]
    fn test_bellstate_collapsed_measurement_00_or_11() {
        let mut reg = QuantumRegister::<f64>::new(2);
        let hadamard = Hadamard::<f64>::new(0);
        let cnot = CNot::new(0, 1);

        reg.add_gate(Box::new(hadamard));
        reg.add_gate(Box::new(cnot));
        reg.run();
        let num_runs = 10000;
        for _ in 0..num_runs {
            let first_bit = reg.measure_with_collapse(0) as u32;
            let second_bit = reg.measure_with_collapse(1) as u32;
            assert!(first_bit == second_bit); // bell state, 1/sqrt(2) * |00> + 1/sqrt(2) * |11>
        }
    }

    #[test]
    fn test_ghz_state_expected_probabilities_f64() {
        // this test doesnt account for collapse right now.
        // because of that the entangled states ghz state doesnt collapse to 000 or 111, but can also become 001, etc.
        // to test the correct behaviour of the collapse there is a different test.
        let mut reg = QuantumRegister::<f64>::new(3);
        let hadamard = Hadamard::<f64>::new(0);
        let cnot = CNot::new(0, 1);
        let cnot2 = CNot::new(1, 2);

        reg.add_gate(Box::new(hadamard));
        reg.add_gate(Box::new(cnot));
        reg.add_gate(Box::new(cnot2));
        reg.run();
        // Check with multiple runs to see if we roughly get expected probabilities
        // This is a probabilistic test. It might fail occasionally, but over a large number of runs,
        // the results should converge to the expected values.
        let num_runs = 10000;
        let mut ones_count_first_bit = 0u32;
        let mut ones_count_second_bit = 0u32;
        let mut ones_count_third_bit = 0u32;
        for _ in 0..num_runs {
            let first_bit = reg.measure_no_collapse(0) as u32;
            let second_bit = reg.measure_no_collapse(1) as u32;
            let third_bit = reg.measure_no_collapse(1) as u32;

            ones_count_first_bit += first_bit;
            ones_count_second_bit += second_bit;
            ones_count_third_bit += third_bit;
        }
        let approx_prob_1_first_bit = ones_count_first_bit as f64 / num_runs as f64;
        let approx_prob_1_second_bit = ones_count_second_bit as f64 / num_runs as f64;
        let approx_prob_1_thrid_bit = ones_count_third_bit as f64 / num_runs as f64;
        assert!((approx_prob_1_first_bit - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
        assert!((approx_prob_1_second_bit - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
        assert!((approx_prob_1_thrid_bit - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
    }

    #[test]
    fn test_ghz_state_collapsed_measurement_000_or_111() {
        let mut reg = QuantumRegister::<f64>::new(3);
        let hadamard = Hadamard::<f64>::new(0);
        let cnot = CNot::new(0, 1);
        let cnot2 = CNot::new(1, 2);

        reg.add_gate(Box::new(hadamard));
        reg.add_gate(Box::new(cnot));
        reg.add_gate(Box::new(cnot2));
        reg.run();
        let num_runs = 10000;
        for _ in 0..num_runs {
            let first_bit = reg.measure_with_collapse(0) as u32;
            let second_bit = reg.measure_with_collapse(1) as u32;
            assert!(first_bit == second_bit);
        }
    }
}
