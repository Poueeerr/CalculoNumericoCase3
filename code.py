import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# GERAÇÃO DE FUNÇÃO QUÁRTICA ALEATÓRIA
# --------------------------

def gerar_funcao_quartica(seed=None):
    """
    Gerador de função quártica (simulação de deflexão da viga)
        - coeficientes
        - função y(x)
        - y'(x)
        - y''(x)
    """
    if seed is not None:
        np.random.seed(seed)

    # Coeficientes em ordem decrescente: a4 x⁴ + a3 x³ + ... + a0
    coef = np.random.uniform(
        low=[-5e-6, -1e-4, 1e-4, -1e-3, -1e-3],
        high=[ 5e-6,  1e-4, 5e-3,  1e-3,  1e-3]
    )

    poly_y = np.poly1d(coef)
    poly_d1 = np.polyder(poly_y)
    poly_d2 = np.polyder(poly_d1)

    return coef, poly_y, poly_d1, poly_d2

# --------------------------
# PASSO 1: Simular dados reais da viga (com função gerada)
# --------------------------

coef_y, y_func, dy_func, d2y_func = gerar_funcao_quartica(seed=42)

# Medições em 20 pontos ao longo da viga de 10 metros
x = np.linspace(0, 10, 20)
y = y_func(x)

# --------------------------
# PASSO 2: Ajustar polinômio de grau 4 aos dados simulados
# --------------------------

ajuste_coef = np.polyfit(x, y, 4)
ajuste_poly = np.poly1d(ajuste_coef)

print("Coeficientes do polinômio ajustado:")
for i, coef in enumerate(ajuste_coef[::-1]):
    print(f"a{i} = {coef:.8f}")

# --------------------------
# PASSO 3: Derivar o polinômio ajustado
# --------------------------

ajuste_d1 = np.polyder(ajuste_poly)
ajuste_d2 = np.polyder(ajuste_d1)

# --------------------------
# PASSO 4: Calcular curvatura e encontrar seu valor máximo
# --------------------------

x_fine = np.linspace(0, 10, 500)
curvatura = ajuste_d2(x_fine)

curv_max = np.max(np.abs(curvatura))
x_curv_max = x_fine[np.argmax(np.abs(curvatura))]

print(f"\n🔍 Curvatura máxima ≈ {curv_max:.6f} m⁻¹")
print(f"📍 Ocorre em x = {x_curv_max:.2f} m")

# --------------------------
# PASSO 5: Visualizar resultados
# --------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'o', label='Medições simuladas')
plt.plot(x_fine, ajuste_poly(x_fine), label='Polinômio Ajustado', color='green')
plt.title("Deflexão simulada e polinômio ajustado")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_fine, curvatura, label="Curvatura ≈ y''(x)", color='blue')
plt.axvline(x_curv_max, color='red', linestyle='--', label=f'Máx. em x = {x_curv_max:.2f}')
plt.title("Curvatura aproximada ao longo da viga")
plt.xlabel("x (m)")
plt.ylabel("Curvatura (1/m)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
