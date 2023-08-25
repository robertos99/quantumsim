use std::ops::Mul;

pub mod cnot;
pub mod cswap;
pub mod hadamard;
pub trait Gate<T>
where
    T: Mul<Output = T> + Copy,
{
    // TODO make this mutate the vector so that we dont waste memory
    fn apply(&self, state_vec: &mut Vec<T>);
}
