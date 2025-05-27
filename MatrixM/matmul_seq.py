#!/usr/bin/env python3
import sys, time
import numpy as np

def main():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} N"); sys.exit(1)
    N = int(sys.argv[1])

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()

    print(f"Matrix {N}x{N} Tiempo: {t1 - t0:.6f} segundos", flush=True)

if __name__ == "__main__":
    main()