#!/usr/bin/env python3

from mpi4py import MPI
import sys, time, math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    if len(sys.argv) != 2:
        if rank == 0:
            print(f"Uso: {sys.argv[0]} D")
        sys.exit(1)

    D = int(sys.argv[1])
    start = 10**(D - 1)
    end = 10**D

    # Calculate the range of numbers each process will handle
    total = end - start
    chunk = total // size
    extra = total % size

    local_start = start + rank * chunk + min(rank, extra) # Offset for extra elements
    local_end = local_start + chunk + (1 if rank < extra else 0)

    t0 = time.perf_counter()
    local_count = sum(1 for i in range(local_start, local_end) if is_prime(i))
    t1 = time.perf_counter()

    # Gather the counts from all processes
    total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"[MPI] Primos con {D} digitos: {total_count}")
        print(f"Tiempo: {t1 - t0:.6f} segundos", flush=True)

    
if __name__ == "__main__":
    main()
