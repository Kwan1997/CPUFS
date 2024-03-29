# CPUFS
Codes for reproducing our paper [*"Unsupervised Feature Selection via Graph Regularized Nonnegative CP Decomposition"*](https://ieeexplore.ieee.org/document/9737358). Specifically, 
- Run the file **"tensorFShandler_guan_penaltyD.m"** for reproducing the results of **CPUFS** on the **ORL** dataset, and 
- Run the file **"tensorFShandler_chen_penaltyD.m"** for reproducing the results of **CPUFSnn** on **the same** dataset.

Required packages:
- **MATLAB Tensor Toolbox** (downloadable from [here](https://www.tensortoolbox.org/)),
- **MATLAB Statistics and Machine Learning Toolbox** (built-in), and
- **MATLAB Parallel Computing Toolbox** (built-in but non-mandatory (but one needs to change "parfor" to simple "for")).

Hope this helps. 😁
