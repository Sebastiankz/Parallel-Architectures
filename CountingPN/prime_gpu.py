#!/usr/bin/env python3

import sys, time, math
import numpy as np
from numba import cuda

@cuda.jit
def mark_primes(start, results):
    i = cuda.grid(1)
    n = start + i

    if i < results.size:
        if n < 2:
            results[i] = 0
            return
        if n == 2:
            results[i] = 1
            return
        if n % 2 == 0:
            results[i] = 0
            return
        
        is_prime = 1
        for d in range(3, int(math.sqrt(n)) + 1, 2):
            if n % d == 0:
                is_prime = 0
                break
        results[i] = is_prime

def main():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} D")
        sys.exit(1)

    D = int(sys.argv[1])
    start = 10**(D - 1)
    end = 10**D
    size = end - start

    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    results_gpu = cuda.device_array(size, dtype=np.int32)

    cuda.synchronize()
    t0 = time.perf_counter()

    mark_primes[blocks, threads_per_block](start, results_gpu)

    cuda.synchronize()
    t1 = time.perf_counter()

    results = results_gpu.copy_to_host()
    count = np.sum(results)

    print(f"[GPU] Primos con {D} digitos: {count}")
    print(f"Tiempo: {t1 - t0:.6f} segundos", flush=True)

if __name__ == "__main__":
    main()

