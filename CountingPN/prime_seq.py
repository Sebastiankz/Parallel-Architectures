#!/usr/bin/env python3
import sys, time
import math

def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} N")
        sys.exit(1)


    D = int(sys.argv[1])
    start = 10**(D - 1)
    end = 10**D

    t0 = time.perf_counter()
    count = sum(1 for i in range(start, end) if is_prime(i))
    t1 = time.perf_counter()

    print(f"[SEQ] Primos de {start} a {end-1}: {count}")
    print(f"Tiempo: {t1 - t0:.6f} segundos", flush=True)

if __name__ == "__main__":
    main()