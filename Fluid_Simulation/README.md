# Fluid Simulation

## Introduction

This application simulates the physics of fluid flows like smoke, clouds, mist, or water flows. The calculation of the flows is done by the Navier-Stokes Equations. The main goals are speed and a convincing look and not physical correctness. The original code is fetched from [Real-Time Fluid Dynamics for Games from Jos Stam](https://github.com/BlainMaguire/3dfluid).

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

Note that the code only supports a cube.

## Optional input parameters

- -h<br/>
  Prints a help for all input parameters.

- -s<br/>
  Enables the check against the sequential computation (This can increase runtime drastically if cube side length or time steps are high).

- -pds<br/>
  Enable printing. This will print the result array of the density buffer of the sequential version to the standard output.

- -pd<br/>
  Enable printing. This will print the result array of the density buffer of the parallel (GPU) version to the standard output.

## Versions

For the Fluid Simulation we provide different implementations for different purposes. Each version takes the same input parameters as mentioned above.

### 3D_Fluid

This implementation computes the same results as the original code. This version includes host-tasks, which leads to a loss of performance compared to the original code.

### 3D_Fluid_Profiling

In the profiling code we analysed the cumulative execution times of all functions.

### 3D_Fluid_Parallel

This version executes all functions on the devices, which leads to slightly different results as the original code but results in better performance.

### 3D_Fluid_Simulation

Provides a 3D simulation of the fluid motion. The code running on the server, which computes the motion for each time step is written for Linux. The client code is written for Windows. Note that the side length of the cube in the client code is defined as a global variable and is not passed as argument.

#### Controls:

- ‘s’ key: insert some particles in the middle of the scene
- ‘x’ key: add force in x direction
- ‘y’ key: add force in y direction
- ‘z’ key: add force in z direction
- left mouse button + movement: rotate the scene

## Contributors

- Pernecker Ralf
- Wald Linus
