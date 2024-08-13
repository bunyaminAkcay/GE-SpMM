# GE-SpMM

This repository contains the implementation of a general-purpose Sparse Matrix-Dense Matrix Multiplication (SpMM) in CUDA C. The project aims to implement and test SpMM kernels based on the pseudocode presented in the article by [Huang et al., 2020].

## Usage

Build project:

```
make
```

Run tests:

```
./build/test
```

Profile project:

```
./profile.sh
```

## References

[1] G. Huang, G. Dai, Y. Wang, and H. Yang, “GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph Neural Networks,” IEEE Xplore, Nov. 01, 2020. https://ieeexplore.ieee.org/abstract/document/9355302/

‌