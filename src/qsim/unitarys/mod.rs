use std::ops::Mul;

pub mod cnot;
pub mod cswap;
pub mod hadamard;
pub trait Gate<T> {
    fn apply(&self, state_vec: &mut Vec<T>);
}
