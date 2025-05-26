import cupy as cp
import sys, time

def main():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} N")
        sys.exit(1)
    N = int(sys.argv[1])

    # Generate random matrices A and B en GPU
    A = cp.random.rand(N, N, dtype=cp.float32)
    B = cp.random.rand(N, N, dtype=cp.float32)

    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    C = A @ B  # o cp.dot(A, B)
    cp.cuda.Device(0).synchronize()
    t1 = time.perf_counter()

    print(f"[GPU] {N}x{N} Tiempo: {t1 - t0:.6f} segundos", flush=True)

if __name__ == "__main__":
    main()

