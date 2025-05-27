import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

matrix_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
prime_digits = [2, 3, 4, 5]
mpi_processes = [1, 2, 4, 6]

matmul_seq = "MatrixM/matmul_seq.py"
matmul_gpu = "MatrixM/matmul_gpu.py"
matmul_mpi = "MatrixM/matmul_mpi.py"

prime_seq = "CountingPN/prime_seq.py"
prime_gpu = "CountingPN/prime_gpu.py"
prime_mpi = "CountingPN/prime_mpi.py"

resultados_matmul = []
resultados_primos = []

def run_script(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True, timeout=300)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Tiempo:" in line:
                    try:
                        return float(line.split("Tiempo:")[1].split()[0])
                    except:
                        pass
        return None
    except subprocess.TimeoutExpired:
        return None

    
for N in matrix_sizes:
    tiempo = run_script(f"python {matmul_seq} {N}")
    resultados_matmul.append({"N": N, "version": "seq", "procesos": 1, "tiempo": tiempo})

    tiempo = run_script(f"python {matmul_gpu} {N}")
    resultados_matmul.append({"N": N, "version": "gpu", "procesos": 1, "tiempo": tiempo})

    for p in mpi_processes:
        tiempo = run_script(f"mpiexec -n {p} python {matmul_mpi} {N}")
        resultados_matmul.append({"N": N, "version": "mpi", "procesos": p, "tiempo": tiempo})

for D in prime_digits:
    tiempo = run_script(f"python {prime_seq} {D}")
    resultados_primos.append({"D": D, "version": "seq", "procesos": 1, "tiempo": tiempo})

    tiempo = run_script(f"python {prime_gpu} {D}")
    resultados_primos.append({"D": D, "version": "gpu", "procesos": 1, "tiempo": tiempo})

    for p in mpi_processes:
        tiempo = run_script(f"mpiexec -n {p} python {prime_mpi} {D}")
        resultados_primos.append({"D": D, "version": "mpi", "procesos": p, "tiempo": tiempo})

# Guardar en CSV
pd.DataFrame(resultados_matmul).to_csv("resultados_matmul.csv", index=False)
pd.DataFrame(resultados_primos).to_csv("resultados_primos.csv", index=False)

print("Resultados guardados en CSV.")

