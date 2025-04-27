import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Funções auxiliares manuais
# --------------------------

# Avaliação de um polinômio de grau n
def avaliar_polinomio(coef, x):
    return sum(c * x**(len(coef)-1-i) for i, c in enumerate(coef))

# Derivada primeira manual de um polinômio
def derivada1(coef, x):
    grau = len(coef) - 1
    return sum((grau - i) * c * x**(grau - i - 1) for i, c in enumerate(coef[:-1]))

# Derivada segunda manual de um polinômio
def derivada2(coef, x):
    grau = len(coef) - 1
    return sum((grau - i)*(grau - i - 1)*c * x**(grau - i - 2) for i, c in enumerate(coef[:-2]))

# Matriz de Vandermonde
def vander(coef, cols=0):
    if not cols:
        cols = len(coef)
    return np.array([[c**(j-1) for j in range(cols, 0, -1)] for c in coef])

# Método de mínimos quadrados manual (sem np.polyfit)
def ajuste_polinomio(x, y, grau):
    A = vander(x, grau + 1)
    coef = np.linalg.solve(A.T @ A, A.T @ y)
    return coef

# --------------------------
# Dados simulados
# --------------------------

# Gerar coeficientes reais para uma função quartica
coef_real = np.random.uniform(low=[-5e-6, -1e-4, 1e-4, -1e-3, -1e-3],
                              high=[ 5e-6,  1e-4, 5e-3,  1e-3,  1e-3])

x = np.linspace(0, 10, 20)
y_real = np.array([avaliar_polinomio(coef_real, xi) for xi in x])

# Curvas para análise
x_fino = np.linspace(0, 10, 500)

# Ajustes para diferentes graus
graus = [3, 4]
curvaturas = {}
erros = {}

for grau in graus:
    # Ajuste manual de polinômio
    coef_fit = ajuste_polinomio(x, y_real, grau)

    # Calcula os valores ajustados
    y_fit = np.array([avaliar_polinomio(coef_fit, xi) for xi in x])

    # Erro absoluto
    erro = y_real - y_fit
    erro_rel = np.abs(erro) / (np.abs(y_real) + 1e-12)

    # Curvatura (segunda derivada) para análise usando o x_fino
    curvatura = np.array([derivada2(coef_fit, xi) for xi in x_fino])
    curv_max = np.max(np.abs(curvatura))
    x_curv_max = x_fino[np.argmax(np.abs(curvatura))]

    curvaturas[grau] = (curvatura, curv_max, x_curv_max)
    erros[grau] = {
        "coef": coef_fit,
        "erro_max": np.max(np.abs(erro)),
        "erro_rel_medio": np.mean(erro_rel),
    }

# --------------------------
# Resultados no console
# --------------------------

print("\n📊 COMPARAÇÃO ENTRE OS AJUSTES:\n")
for grau in graus:
    print(f"--- Polinômio grau {grau} ---")
    print("  Coeficientes:", np.round(erros[grau]["coef"], 8))
    print(f"  Erro absoluto máximo: {erros[grau]['erro_max']:.4e}")
    print(f"  Erro relativo médio:  {erros[grau]['erro_rel_medio']:.4%}")
    print(f"  Curvatura máxima:    {curvaturas[grau][1]:.6f}  em x = {curvaturas[grau][2]:.2f}")
    print()

# --------------------------
# Visualização
# --------------------------

plt.figure(figsize=(18, 5))  # Aumenta a largura para 3 gráficos lado a lado

# 1. Deflexão simulada e ajuste
plt.subplot(1, 3, 1)
plt.plot(x, y_real, 'o', label='Medições simuladas', color='orange')
for grau in graus:
    y_fit = np.array([avaliar_polinomio(erros[grau]["coef"], xi) for xi in x])
    plt.plot(x, y_fit, label=f"Polinômio Ajustado (grau {grau})")
plt.title("Deflexão simulada e polinômio ajustado")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.legend()

# 2. Curvatura ao longo da viga
plt.subplot(1, 3, 2)
for grau in graus:
    curv, max_c, x_c = curvaturas[grau]
    plt.plot(x_fino, curv, label=f"Curvatura grau {grau}")
    plt.axvline(x_c, linestyle='--', alpha=0.7, label=f"Máx. em x = {x_c:.2f}")
plt.title("Curvatura aproximada ao longo da viga")
plt.xlabel("x (m)")
plt.ylabel("Curvatura (1/m)")
plt.grid(True)
plt.legend()

# 3. Erro absoluto
plt.subplot(1, 3, 3)
for grau in graus:
    y_fit = np.array([avaliar_polinomio(erros[grau]["coef"], xi) for xi in x])
    erro_abs = np.abs(y_real - y_fit)
    plt.plot(x, erro_abs, 'o--', label=f"Erro grau {grau}")
plt.title("Erro absoluto nas medições")
plt.xlabel("x (m)")
plt.ylabel("|y_real - y_fit|")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
