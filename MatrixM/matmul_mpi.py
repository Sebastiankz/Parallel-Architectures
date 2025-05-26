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


    counts = [N // size + (1 if i < N % size else 0) for i in range(size)]
    displs = [sum(counts[:i]) for i in range(size)]

    local_row = counts[rank] # Number of rows for this process

    A_part = np.empty((local_row, N), dtype=np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C_part = np.empty((local_row, N), dtype=np.float32)

    if rank == 0:
        A = np.random.rand(N, N).astype(np.float32)
        B[:] = np.random.rand(N, N).astype(np.float32)
    else:
        A = None

    sendCounts = [r * N for r in counts] # Send counts in bytes
    displs_bytes = [d * N for d in displs] # Displacements in bytes
    comm.Scatterv([A, sendCounts, displs_bytes, MPI.FLOAT], A_part, root=0)

    comm.Bcast([B, MPI.FLOAT], root=0)

    t0 = time.perf_counter()
    C_part[:] = A_part @ B
    t1 = time.perf_counter()

    if rank == 0:
        C = np.empty((N, N), dtype=np.float32)
    else:
        C = None

    recvcounts = [r * N for r in counts]
    recvdispls = [d * N for d in displs]
    comm.Gatherv(C_part, [C, recvcounts, recvdispls, MPI.FLOAT], root=0)

    if rank == 0:
        print(f"[MPI] {N}x{N} Tiempo: {t1 - t0:.6f} segundos", flush=True)
    
if __name__ == "__main__":
    main()




    

    



