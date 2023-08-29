# About
This is my approach in building a quantum computation simulator.
This is mostly a project to teach me quantum computation and rust.
For now im only using amplitudes with real parts. Will add complex amplitudes later.
Will also add less memory intensive options like f32 based registers and gates.


# Todo's
- Make the gates less memory intensive. Right now most gates create a full fledged matrix for rotating the state vetor. This leads to a overall space complexity of $2^n$ for the state vec itself and a $2^{n^2}$ for the matrix representing the gate and an additional $2^n$ for the new intermediate state vec that is rotated. Leading to a total space complexitity of $2^n$ + $2^n$ + $2^{n^2}$. This can be reduced to $2^n$ + $2^n$ + $2^n$ or maybe just $2^n$ + $2^n$ by doing the matrix calculation row by row without creating the full matrix beforehand. For now im not using large registers and its just much simpler to think about the rotation as the matrix. Also easier to verify/test. 

- calculate required memory consumption and pre allocate so we have less system calls. this would also be great to calculate required worker node size. 

- Add more gates. Duh.

- Add classical gates? Maybe not. Maybe this would be a seperate crate to keep this part for pure states and no logical classical decisions.

- Error correction and simulating it in the first place.
