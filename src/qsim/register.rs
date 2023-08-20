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

impl<T> QuantumRegister<T>
where
    T: Mul<Output = T> + Copy + Magnitude<Output = f64> + Zero + One,
{
    pub fn new(amount_qubits: usize) -> Self {
        // This is simply to prevent me from creating registers that become too large.
        // Can be adjusted as desired.
        assert!(
            amount_qubits < 8,
            "Bit index {} too large. Maximum allowed is 8.",
            amount_qubits
        );

        let mut init = vec![T::zero(); 2_usize.pow(amount_qubits as u32)];
        init[0] = T::one();
        QuantumRegister {
            state_vector: init,
            gates: vec![],
        }
    }

    pub fn add_gate(&mut self, gate: Box<dyn Gate<T>>) {
        self.gates.push(gate);
    }

    pub fn run(&mut self) {
        for gate in &self.gates {
            let new_state_vec = gate.apply(&self.state_vector);
            self.state_vector = new_state_vec;
        }
    }

    pub fn measure(&self, i: usize) -> u8 {
        // Register is 0 indexed.
        assert!(
            i < get_amount_bits(&self.state_vector),
            "i: {} is too large. The register holds {} Bits and is indexed 0.",
            i,
            get_amount_bits(&self.state_vector)
        );

        let mut prob_1 = 0_f64;
        for (index, value) in self.state_vector.iter().enumerate() {
            // Bitshifting to check if the index of current amplitude corresponds with a 0 or 1 bit for the i'th Qubit.
            // i = 0 for the left most bit in the register. Example: if the register holds |10> then i = 0 points to the Qubit that is in state 1.
            // index(4)                     = 0000 0100
            // mask amount_qbits(4) - i(0)  = 0000 0100
            if index & 1 << get_amount_bits(&self.state_vector) - 1 - i != 0 {
                // sum of square of magnitude
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
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::qsim::{math::Complex, unitarys::CNot, unitarys::Hadamard};

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
        let reg = QuantumRegister {
            state_vector: vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0, 0.0],
            gates: vec![],
        };

        // Check the measure function returns a valid value
        let result = reg.measure(0);
        assert!(result == 0 || result == 1);
    }

    #[test]
    #[should_panic(expected = "i: 3 is too large. The register holds 2 Bits and is indexed 0.")]
    fn test_measure_out_of_bounds_assertion() {
        let reg = QuantumRegister {
            state_vector: vec![1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0, 0.0],
            gates: vec![],
        };

        // This should panic and thus pass the test. register holds 2 bits
        reg.measure(3);
    }

    #[test]
    fn test_measure_expected_probabilities() {
        // equivilant to |+0>
        let reg = QuantumRegister {
            state_vector: vec![1.0 / 2.0f64.sqrt(), 0.0, 1.0 / 2.0f64.sqrt(), 0.0],
            gates: vec![],
        };

        // Check with multiple runs to see if we roughly get expected probabilities
        // This is a probabilistic test. It might fail occasionally, but over a large number of runs,
        // the results should converge to the expected values.
        let num_runs = 10000;
        let mut ones_count = 0u32;
        for _ in 0..num_runs {
            ones_count += reg.measure(0) as u32;
        }
        let approx_prob_1 = ones_count as f64 / num_runs as f64;
        assert!((approx_prob_1 - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
    }

    #[test]
    fn test_create_bellstate_expected_probabilities_f64() {
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
        let mut ones_count = 0u32;
        for _ in 0..num_runs {
            let first_bit = reg.measure(0) as u32;
            let second_bit = reg.measure(1) as u32;
            assert!(first_bit == second_bit); // bell state, 1/sqrt(2) * |00> + 1/sqrt(2) * |11>

            ones_count += first_bit;
        }
        let approx_prob_1 = ones_count as f64 / num_runs as f64;
        assert!((approx_prob_1 - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
    }

    #[test]
    fn test_create_bellstate3_expected_probabilities_f64() {
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
        let mut ones_count = 0u32;
        for _ in 0..num_runs {
            let first_bit = reg.measure(0) as u32;
            let second_bit = reg.measure(1) as u32;
            assert!(first_bit == second_bit); // bell state, 1/sqrt(2) * |00> + 1/sqrt(2) * |11>

            ones_count += first_bit;
        }
        let approx_prob_1 = ones_count as f64 / num_runs as f64;
        assert!((approx_prob_1 - 0.5).abs() < 0.05); // Check if it's close to 0.5 within a margin
    }
}
