# Memory Copy

## Introduction

The Memory Copy is a simple benchmark for checking the GPU memory bandwidth. Before starting the time measurement the data is prefetched to the GPU.

## Prerequisites

- Boost 1.68
- OpenMPI 4.0.0
- HipSYCL
- C++ 17 compiler
- Celerity

## Optional input parameters

- -s<br/>
  Specify seed for random number generator (standard value: 1)

- -m<br/>
  Specify memory size to copy (standard value: 7e8)

## Contributors

- Pernecker Ralf
- Wald Linus
