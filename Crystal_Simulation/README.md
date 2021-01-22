# Crystal Simulation

## Introduction

The Crystal Simulation is an application to determine specifications of crystals. In crystallography it is used to identify the exact shapes of a molecule with X-ray diffraction on single crystals or powders The first version is a straightforward implementation regarding the buffer access pattern, leading to a divergent control flow in the kernel code. The second approach rearranges the data to omit a divergent control flow, which avoids a masking by the hardware and leads to better performance. The original code is fetched from a collection of exercises for using OmpSs from the [Barcelona Supercomputing Center](https://github.com/bsc-pm/ompss-ee/tree/master/03-gpu-devices/krist-opencl).

## Prerequisites

- Boost 1.68
- OpenMPI 4.0.0
- HipSYCL
- C++ 17 compiler
- Celerity

## Required input parameters

Expected input: `./[file_name] atoms reflections [optional parameters]`

1. atoms: defines the number of atoms
2. reflections: defines the number of reflections

## Optional input parameters

- -s<br/>
  Enables the check against the sequential computation (This can increase runtime drastically if atoms and reflections are high).

- -p<br/>
  Enable printing. This will print the result array to the standard output. If the sequential check is enabled both arrays will be printed. The print will only be executed if the verification is successful.

- -pr<br/>
  Enable progression. This will print the current progression of the sequential computation. If enabled the time measurement for the sequential computation will be affected.

## Contributors

- Pernecker Ralf
- Wald Linus
