# 3D Heat Stencil

## Introduction

The 3D Heat Stencil code calculates the heat distribution in a 3D domain. The heat source is a fixed point, which remains on the same temperature for the whole calculation. To evaluate the new temperature value for a given point only the direct neighbours are considered. The original code is fetched from [Introduction to parallel computing and parallel algorithms](https://github.com/philippgs/uibk_ipcpa_19).

## Prerequisites

- Boost 1.68
- OpenMPI 4.0.0
- HipSYCL
- C++ 17 compiler
- Celerity

## Required input parameters

Expected input: `./[file_name] timesteps height width depth [optional parameters]`

1. timesteps: defines the number of timesteps executed
2. height: defines the height of the cuboid
3. width: defines the width of the cuboid
4. depth: defines the depth of the cuboid

## Optional input parameters

- -s<br/>
  Enables the check against the sequential computation (This can increase runtime drastically if height, width, depth and time steps are high).

- -p<br/>
  Enable printing. This will print the result array to the standard output. If the sequential check is enabled both arrays will be printed. The print will only be executed if the verification is successful.

- -pr<br/>
  Enable progression. This will print the current progression of the sequential computation. If enabled the time measurement for the sequential computation will be affected.

## Contributors

- Pernecker Ralf
- Wald Linus
