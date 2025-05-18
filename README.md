# LOLA-FHE: Fast FHE Implementation of LOLA (Low latency CKKS bootstrapping for few slots)

This repository contains implementations of Fully Homomorphic Encryption (FHE) schemes in SageMath.
It serves as a proof-of-concept implementation for the paper "Low latency bootstrapping for CKKS using Roots of Unity" by Coron & Köstler see [LOLA paper](https://eprint.iacr.org/2025/651).
For comparison, we also provide an implementation of the CKKS scheme using the same SageMath framework.
The code provides efficient implementations for both clear-text and homomorphic DFT operations (SlotToCoeff and CoeffToSlot), along with bootstrapping capabilities for FHE schemes.

## Repository Structure

- `DFT_class.sage`: Implementation of the Fast DFT module for clear-text operations
- `FHE_DFT_class.sage`: Homomorphic DFT implementation for FHE operations
- `lola.ipynb`: Implementation and testing of bootstrapping for the LOLA scheme
- `lola.sage`: SageMath file for imports
- `ckks.ipynb`: Implementation and testing of bootstrapping for CKKS scheme
- `ckks.sage`: SageMath file for imports
- `fastDFT.ipynb`: Notebook generating and testing the Fast DFT implementation, it can write the .sage files
- `root_of_unity.sage`: Helper functions for root of unity calculations
- `sagefhepoly/`: Submodule containing polynomial arithmetic implementations

## Features

- Fast DFT implementation with both normal and bit-reversed order operations
- Homomorphic DFT operations for FHE schemes, without the radix decomposition
- Bootstrapping implementations for both the LOLA and the CKKS scheme
- Runtimes (for few slots) and ring dimension 2^15 are in the order of seconds
- Efficient polynomial arithmetic in the ring Z_q[X]/(X^N+1) through the submodule `sagefhepoly/`
- Support for various precision levels and parameter configurations

## Requirements

- SageMath 10.2 or later
- Python 3.11 or later

## Usage

1. First, ensure you have SageMath installed and the required dependencies.

2. Play around in the notebook files!

## Parameters

- `N`: Ring dimension (must be a power of 2)
- `slots`: Number of slots for encoding (must be ≤ N/2)
- `modulus`: Ciphertext (base) modulus
- `precision`: Precision for floating-point operations, which determines the scaling factor, often Delta
- `d`: Degree of Taylor approximation (for CKKS)
- `r`: Number of squarings (for CKKS)

## Notes

- The implementation supports both normal and bit-reversed order operations for the DFT operations.
- The code includes various precomputation steps for improved performance
- The implementation is not optimized for performance, but for readability and ease of use.
It does not comprise an RNS implementation.
All heavy computations are outsourced to SageMath's underlying polynomial arithmetic supplied by the NTL library.
For more details on the polynomial arithmetic, see the submodule `sagefhepoly/`.
Since some basic Python operations are used in the code, one may argue that the performance comparison does not reflect the performance of the actual implementation.
However, this is not the case, since the proportion of the latter operation is negligible compared to the heavy operations (e.g. multiplications) executed in the NTL library.

## License

MIT License

Copyright (c) 2025 Robin Koestler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:


