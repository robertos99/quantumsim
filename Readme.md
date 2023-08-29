# About
This is my approach in building a quantum computation simulator.
This is mostly a project to teach me quantum computation and rust.
For now im only using amplitudes with real parts. Will add complex amplitudes later.
Will also add less memory intensive options like f32 based registers and gates.


# Todo's
1. Make the gates less memory intensive. Right now most gates create a full fledged matrix for rotating the state vetor. This leads to a overall space complexity of $2^n$ for the state vec itself and a $2^{n^2}$ for the matrix representing the gate. Leading to a total space complexitity of $2^n$ + $2^{n^2}$.

2. Add more gates. Duh.

3. Add classical gates? Maybe not. Maybe this would be a seperate crate to keep this part for pure states and no logical classical decisions.

4. Error correction and simulating it in the first place.