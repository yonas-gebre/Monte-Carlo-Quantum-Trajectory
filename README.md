# Monte-Carlo-Quantum-Trajectory
Fully parallelized Monte Carlo simulation for modeling the ionization of atoms interacting with intense laser pulses.
The model uses the PPT model of ionization to initialize a Monte Carlo simulation over electron trajectories that are treated semi-classically. We use the Feynman method of calculating the phase to allow for interference effects to take place.

1. [ Dependencies. ](#desc)
2. [ Usage. ](#usage)
3. [ Credits. ](#development)


<a name="desc"></a>
## 1. Dependencies

matplotlib, scipy,  h5py, mpi4py

<a name="usage"></a>
## 2. Usage

The code can be run by using the TISE.py file inside the directory source_code. To run with 12 processors one writes:

mpiexec -n 12 python Classical_Trajectory.py

This will output a set of file in a subdirectory called "Data". Each file corresponds to an output from each processors.
The parameters are set in the Classical_Trajectory.py file in the current version. Also included are some analysis scripts that will have documentation very soon.


<a name="development"></a>
## 3. Credits
This project was done with work funded by the  National Science Foundation (NSF) and the U.S. Department of Energy. It was conducted under the guidance of Andreas Becker, a research professor at JILA, University of Colorado Boulder.