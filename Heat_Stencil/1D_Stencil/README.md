# 1D Heat Stencil

## Introduction

The 1D Heat Stencil code calculates the heat distribution in a 1D domain. The heat source is a fixed point, which remains on the same temperature for the whole calculation. To evaluate the new temperature value for a given point only the direct neighbours are considered. The original code is fetched from [Introduction to parallel computing and parallel algorithms](https://github.com/philippgs/uibk_ipcpa_19).

## Prerequisites

- Boost 1.68
- OpenMPI 4.0.0
- HipSYCL
- C++ 17 compiler
- Celerity

## Required input parameters

Expected input: `./[file_name] timesteps length [optional parameters]`

1. timesteps: defines the number of timesteps executed
2. length: defines the length of the stencil

## Optional input parameters

- -s<br/>
  Enables the check against the sequential computation (This can increase runtime drastically if length and time steps are high).

- -d<br/>
  Enable simulation. After each 100 time steps the current heat distribution is printed to the standard output.

- -p<br/>
  Enable printing. This will print the result array to the standard output. If the sequential check is enabled both arrays will be printed. The print will only be executed if the verification is successful.

- -pr<br/>
  Enable progression. This will print the current progression of the sequential computation. If enabled the time measurement for the sequential computation will be affected.

## Contributors

- Pernecker Ralf
- Wald Linus
