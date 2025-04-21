import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Funções auxiliares (avaliação e derivadas manuais)
# --------------------------

def gerar_funcao_quartica(seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(
        low=[-5e-6, -1e-4, 1e-4, -1e-3, -1e-3],
        high=[ 5e-6,  1e-4, 5e-3,  1e-3,  1e-3]
    )

def avaliar_polinomio(coef, x):
    return sum(c * x**p for c, p in zip(coef, range(len(coef)-1, -1, -1)))

def derivada1(coef, x):
    grau = len(coef) - 1
    return sum((grau - i) * c * x**(grau - i - 1) for i, c in enumerate(coef[:-1]))

def derivada2(coef, x):
    grau = len(coef) - 1
    return sum((grau - i)*(grau - i - 1)*c * x**(grau - i - 2)
               for i, c in enumerate(coef[:-2]))

# --------------------------
# Dados simulados
# --------------------------

coef_real = gerar_funcao_quartica(seed=42)
x = np.linspace(0, 10, 20)
y_real = avaliar_polinomio(coef_real, x)

# Curvas para análise
x_fino = np.linspace(0, 10, 500)
curv_real = derivada2(coef_real, x_fino)
curvaturas = {}
erros = {}

# --------------------------
# Ajustes para diferentes graus
# --------------------------

graus = [3, 4, 5]

for grau in graus:
    coef_fit = np.polyfit(x, y_real, grau)
    y_fit = avaliar_polinomio(coef_fit, x)
    erro = y_real - y_fit
    erro_rel = np.abs(erro) / (np.abs(y_real) + 1e-12)

    curv_fit = derivada2(coef_fit, x_fino)
    max_curv = np.max(np.abs(curv_fit))
    x_max = x_fino[np.argmax(np.abs(curv_fit))]

    curvaturas[grau] = (curv_fit, max_curv, x_max)
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

plt.figure(figsize=(14, 6))

# Plot da curvatura real e ajustadas
plt.subplot(1, 2, 1)
plt.plot(x_fino, curv_real, label="Curvatura Real (grau 4)", linewidth=2)
for grau in graus:
    curv, max_c, x_c = curvaturas[grau]
    plt.plot(x_fino, curv, '--', label=f"Ajuste grau {grau}")
    plt.axvline(x_c, linestyle=':', alpha=0.5)

plt.title("Comparação das Curvaturas")
plt.xlabel("x (m)")
plt.ylabel("Curvatura (1/m)")
plt.grid(True)
plt.legend()

# Plot de erro absoluto
plt.subplot(1, 2, 2)
for grau in graus:
    y_fit = avaliar_polinomio(erros[grau]["coef"], x)
    erro_abs = np.abs(y_real - y_fit)
    plt.plot(x, erro_abs, 'o--', label=f"Erro grau {grau}")

plt.title("Erro absoluto nas medições")
plt.xlabel("x (m)")
plt.ylabel("|y_real - y_fit|")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
