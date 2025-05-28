import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from time import perf_counter

# Nuevas configuraciones de pruebas
matrix_sizes = list(range(100, 5001, 100))  # Más puntos: N = 100, 200, ..., 5000
prime_digits = [2, 3, 4, 5, 6]  # Más puntos: D = 2, 3, 4, 5, 6, 8, 10

matmul_seq = "MatrixM/matmul_seq.py"
matmul_gpu = "MatrixM/matmul_gpu.py"
matmul_mpi = "MatrixM/matmul_mpi.py"

prime_seq = "CountingPN/prime_seq.py"
prime_gpu = "CountingPN/prime_gpu.py"
prime_mpi = "CountingPN/prime_mpi.py"

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
        print(f"Timeout en comando: {command}")
        return None

def obtener_mejor_proceso(script_path, parametro, valores_procesos):
    tiempos = {}
    for p in valores_procesos:
        tiempo = run_script(f"mpiexec -n {p} python {script_path} {parametro}")
        if tiempo is not None:
            tiempos[p] = tiempo
    if tiempos:
        return min(tiempos, key=tiempos.get)
    return None

# Determinar procesos ideales
print("Buscando número ideal de procesos para matmul y primos...")
ideal_proc_matmul = obtener_mejor_proceso(matmul_mpi, 10000, [1, 2, 4, 6])
ideal_proc_primos = obtener_mejor_proceso(prime_mpi, 4, [1, 2, 4, 6])
print(f"Ideal matmul: {ideal_proc_matmul} procesos")
print(f"Ideal primos: {ideal_proc_primos} procesos")

# ----------------------
# Graficar MATRICES
# ----------------------
matmul_data = {"N": [], "seq": [], "mpi": [], "gpu": []}

for N in matrix_sizes:
    matmul_data["N"].append(N)
    matmul_data["seq"].append(run_script(f"python {matmul_seq} {N}"))
    matmul_data["gpu"].append(run_script(f"python {matmul_gpu} {N}"))
    matmul_data["mpi"].append(run_script(f"mpiexec -n {ideal_proc_matmul} python {matmul_mpi} {N}"))

df_mat = pd.DataFrame(matmul_data)

plt.figure(figsize=(8,6))
for col in ['seq', 'mpi', 'gpu']:
    plt.plot(df_mat['N'], df_mat[col], linestyle='-', label=col.upper())  # sin marker

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Tamaño de matriz N")
plt.ylabel("Tiempo (s)")
plt.title("Tiempo vs N - Multiplicación de matrices")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.savefig("grafica_matmul_final.png")

# ----------------------
# Graficar PRIMOS
# ----------------------
primo_data = {"D": [], "seq": [], "mpi": [], "gpu": []}

for D in prime_digits:
    primo_data["D"].append(D)
    primo_data["seq"].append(run_script(f"python {prime_seq} {D}"))
    primo_data["gpu"].append(run_script(f"python {prime_gpu} {D}"))
    primo_data["mpi"].append(run_script(f"mpiexec -n {ideal_proc_primos} python {prime_mpi} {D}"))

df_primos = pd.DataFrame(primo_data)

plt.figure(figsize=(8,6))
for col in ['seq', 'mpi', 'gpu']:
    plt.plot(df_primos['D'], df_primos[col], marker='o', label=col.upper())

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Dígitos (D")
plt.ylabel("Tiempo (s)")
plt.title("Tiempo vs D - Conteo de primos")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.savefig("grafica_primos_final.png")

print("Gráficas finalizadas y guardadas: grafica_matmul_final.png y grafica_primos_final.png")
