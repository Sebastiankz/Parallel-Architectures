# Parallel-Architectures

Explorar arquitecturas paralelas

Instalar paquetes
pip install numpy mpi4py cupy-cuda12x

Para ejecutar archivos

1.matmul_seq:
python matmul_seq.py N

2. matmul_mpi:
   mpiexec -n 4 python matmul_mpi.py 512

donde 4 es la cantidad de procesos y 512 la cantidad de filas de la matriz (modificables)
