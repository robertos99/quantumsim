mod qsim;

use qsim::unitarys::{ccnot::CCNot, cswap::CSwap, hadamard::Hadamard, not::Not};

use crate::qsim::register::QuantumRegister;
fn main() {
    // This is a more or less useless "quantum circuit".
    // The idea here is to show a simple circuit that can identify |110> and |011> with 100% probality.
    // This should also show that if we put in a uniform superposition |+++>, by linearity we should
    // get a readout proabability of 1 thats equal to 2/8. This is due to the fact that in the superposition
    // 2/8 of the options are of the desired pattern.

    // We use a register with 8 bits.
    // Wven tho we only have 3 bits we want to check, each check uses 2 ancillas.
    // The last output bit at index 7, so the 8th bit is an additional output bit.
    // While this large amount of bits isnt required it makes things easier to reason about.
    let mut reg = QuantumRegister::new(8);
    // lets first create the 110 state out of the 000.. state in the initialized register
    reg.add_gate(Box::new(Not::new(0)));
    reg.add_gate(Box::new(Not::new(1)));
    // the state 110 for the 3 bits that we test is now prepared.

    // checking for 110
    reg.add_gate(Box::new(CCNot::new(0, 1, 3)));
    reg.add_gate(Box::new(Not::new(2)));
    reg.add_gate(Box::new(CCNot::new(2, 3, 4)));
    //resetting the 3rd bit to its original state
    reg.add_gate(Box::new(Not::new(2)));

    // checking for 011
    reg.add_gate(Box::new(CCNot::new(1, 2, 5)));
    reg.add_gate(Box::new(Not::new(0)));
    reg.add_gate(Box::new(CCNot::new(0, 5, 6)));

    // Now the ancillas 6 and 7 at index 5 and 6 respetively should reflect if one of the states is 110 or 011.
    // The final check is if either 6 or 7 is 1. Basicly OR.

    reg.add_gate(Box::new(Not::new(7)));
    reg.add_gate(Box::new(CSwap::new(4, 6, 7)));
    reg.run();

    let result = reg.measure_no_collapse(6);

    println!("The result for the state |110> was {result}.");
    run_for_uniform_superposition();
}

fn run_for_uniform_superposition() {
    let mut reg = QuantumRegister::new(8);
    // lets first create the 110 state out of the 000.. state in the initialized register
    reg.add_gate(Box::new(Hadamard::<f64>::new(0)));
    reg.add_gate(Box::new(Hadamard::<f64>::new(1)));
    reg.add_gate(Box::new(Hadamard::<f64>::new(2)));
    // the state 110 for the 3 bits that we test is now prepared.

    // checking for 110
    reg.add_gate(Box::new(CCNot::new(0, 1, 3)));
    reg.add_gate(Box::new(Not::new(2)));
    reg.add_gate(Box::new(CCNot::new(2, 3, 4)));
    //resetting the 3rd bit to its original state
    reg.add_gate(Box::new(Not::new(2)));

    // checking for 011
    reg.add_gate(Box::new(CCNot::new(1, 2, 5)));
    reg.add_gate(Box::new(Not::new(0)));
    reg.add_gate(Box::new(CCNot::new(0, 5, 6)));

    // Now the ancillas 6 and 7 at index 5 and 6 respetively should reflect if one of the states is 110 or 011.
    // The final check is if either 6 or 7 is 1. Basicly OR.

    reg.add_gate(Box::new(Not::new(7)));
    reg.add_gate(Box::new(CSwap::new(4, 6, 7)));
    reg.run();

    let num_runs = 10000u32;
    let mut num_ones = 0u32;
    for _ in 0..num_runs {
        let result = reg.measure_no_collapse(6);
        num_ones += result as u32;
    }

    let measured_probability = num_ones as f64 / num_runs as f64;
    print!("The measured probability for the uniform superposition as input was {measured_probability}");
}
