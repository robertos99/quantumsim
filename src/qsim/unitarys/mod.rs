pub mod ccnot;
pub mod cnot;
pub mod cswap;
pub mod hadamard;
pub mod not;
pub trait Gate<T> {
    fn apply(&self, state_vec: &mut Vec<T>);
}
