# E^3FS: Efficient, Secure, and Verifiable Fuzzy Search with Data Updates in Hybrid-Storage Blockchains

## Artifact Dependencies and Requirements
**Hardware resources required:** An x64 system with 16GB free memory.

**Operating systems required:** Windows/Linux.

**Software libraries needed:** Python, NumPy, pypbc, sklearn, matplotlib, etc.

## Reproducibility of Experiments
**Algorithms Implemented in This Package:**
1. **E^3FS:** Our proposed algorithm for Hybrid-Storage blockchains.
2. **VFSA:** A baseline algorithm for encrypted databases from [Li 23](https://ieeexplore.ieee.org/abstract/document/9669122).
3. **VRMFS:** A baseline algorithm for encrypted databases from [Tong 23](https://ieeexplore.ieee.org/document/9714876).
4. **MFS:** A baseline algorithm for encrypted databases from [Wang 14](https://ieeexplore.ieee.org/document/6848153).

I've simply adjusted the formatting to make the algorithm names and descriptions stand out more clearly.

## Complete Description of Packages

**dataset**
A real-world medical dialogue dataset from [Chen 20](https://arxiv.org/pdf/2004.03329v1).

**E^3FS**

Contains the following programs:

- `Genkey.py`: Program to generate the secret keys for the scheme.
- `fuzzy_process.py`: Program to read files and create fuzzy keywords.
- `BF_process.py`: Program to create a forward index for each file.
- `ABBT.py`: Program to create an A-BBT for each keyword.
- `scheme.py`: Program to implementate an E3FS scheme.

The algorithm implementation of the baseline scheme can be found in each algorithm package, such as `VFSA`, `VRMFS`, and `MFS`. All protocols can be directly executed by running `scheme.py`.

