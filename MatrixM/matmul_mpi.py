#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import sys, time

comm = MPI.COMM_WORLD  # Create a communicator object
rank = comm.Get_rank() # Get the rank of the current process
size = comm.Get_size() # Get the total number of processes

def main():
    if len(sys.argv) != 2:
        if rank == 0:
            print(f"Uso: {sys.argv[0]} N")
        sys.exit(1)
    N = int(sys.argv[1])


    # Initialize matrices
    rows_per_process = N // size # Number of rows each process will handle
    A_part = np.empty((rows_per_process, N), dtype=np.float32) # Matrix A part
    B = np.empty((N, N), dtype=np.float32) # Matrix B
    C_part = np.empty((rows_per_process, N), dtype=np.float32) # Result matrix part

    # Broadcast the size of the matrix and initialize A and B
    if rank == 0:
        A = np.random.rand(N, N).astype(np.float32)
        B[:] = np.random.rand(N, N).astype(np.float32)
    else:
        A = None

    # Scatter A and broadcast B
    comm.Scatter([A, MPI.FLOAT], A_part, root=0)  # Scatter A to all processes
    comm.Bcast([B, MPI.FLOAT], root = 0)  # Broadcast B to all processes

    t0 = time.perf_counter()  # Start timer
    C_part[:] = A_part @ B  # Perform local matrix multiplication
    t1 = time.perf_counter()  # End timer

    if rank == 0:
        C = np.empty((N, N), dtype=np.float32)
    else:
        C = None

    comm.Gather(C_part, C, root=0) 

    if rank == 0:
       print(f"[MPI] {N}x{N} Tiempo: {t1 - t0:.6f} segundos", flush=True)

if __name__ == "__main__":
    main()




    

    



