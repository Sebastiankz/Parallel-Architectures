# Parallel-Architectures

Explorar arquitecturas paralelas

Instalar paquetes
pip install numpy mpi4py cupy-cuda12x

- Para ejecutar archivos de MatrixM

  1.matmul_seq:
  python matmul_seq.py N

2. matmul_mpi: Necesario instalar Microsoft MPI
   mpiexec -n 4 python matmul_mpi.py 512
   donde 4 es la cantidad de procesos y 512 la cantidad de filas de la matriz (modificables)

3. mamtul_gpu: Necesario instalar Nvidia CUDA (version 12.x)
   python matmul_gpu.py 512

- Para ejecutar archivos de CountingPN

1. countPn_seq:
