"""
graficos.py — Gráficos de resultado: curvas de custo/métrica, comparações,
matriz de confusão, real × previsto e tabela final.
(Histogramas e normas de gradiente estão em graficos_gradientes.py.)

Convenções de cor:
  * treino = azul, validação = laranja (sempre as mesmas cores);
  * camadas = tons de azul do claro (1ª camada) ao escuro (última),
    porque "profundidade" é uma grandeza ordenada.
"""
import numpy as np
import matplotlib.pyplot as plt

CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COR_TREINO = CORES[0]
COR_VAL = CORES[1]
RAMPA_AZUL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
NOMES_METRICA = {"r2": "R²", "acuracia": "Acurácia"}


def configurar_estilo():
    """Visual limpo: grade discreta, sem bordas de cima/direita, linhas finas."""
    plt.rcParams.update({
        "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8984", "axes.titlesize": 11, "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 9,
    })


def cor_da_camada(indice, n_camadas):
    """Tom de azul proporcional à profundidade (0 = primeira camada)."""
    posicao = 0 if n_camadas == 1 else indice / (n_camadas - 1)
    return RAMPA_AZUL[int(round(posicao * (len(RAMPA_AZUL) - 1)))]


def plot_curvas(historico, titulo=""):
    """Custo vs. época e métrica vs. época, treino e validação."""
    epocas = np.arange(1, len(historico["perda_treino"]) + 1)
    nome_metrica = NOMES_METRICA.get(historico["metrica"], historico["metrica"])
    fig, (ax_perda, ax_metrica) = plt.subplots(1, 2, figsize=(11, 3.6))

    ax_perda.plot(epocas, historico["perda_treino"], color=COR_TREINO, label="treino")
    ax_perda.plot(epocas, historico["perda_val"], color=COR_VAL, label="validação")
    ax_perda.set(title="Custo por época", xlabel="época", ylabel="custo")
    ax_metrica.plot(epocas, historico["metrica_treino"], color=COR_TREINO, label="treino")
    ax_metrica.plot(epocas, historico["metrica_val"], color=COR_VAL, label="validação")
    ax_metrica.set(title=f"{nome_metrica} por época", xlabel="época", ylabel=nome_metrica)
    if historico.get("melhor_epoca"):  # linha pontilhada: melhor época (early stopping)
        for ax in (ax_perda, ax_metrica):
            ax.axvline(historico["melhor_epoca"], color="#8a8984", linestyle=":", linewidth=1)
    ax_perda.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    plt.show()


def plot_comparacao(historicos, chave="perda_val", titulo="", ylabel=""):
    """Uma curva por configuração (ex.: SGD × Adam) para a mesma chave do histórico."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (nome, historico) in enumerate(historicos.items()):
        valores = historico[chave]
        ax.plot(np.arange(1, len(valores) + 1), valores, color=CORES[i], label=nome)
    ax.set(title=titulo, xlabel="época", ylabel=ylabel or chave)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_matriz_confusao(predicao, alvo, nomes):
    """Linhas = classe verdadeira, colunas = classe prevista (contagens)."""
    n = len(nomes)
    matriz = np.zeros((n, n), dtype=int)
    for verdadeira, prevista in zip(np.argmax(alvo, axis=1), np.argmax(predicao, axis=1)):
        matriz[verdadeira, prevista] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matriz, cmap="Blues")
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            cor_texto = "white" if matriz[i, j] > matriz.max() / 2 else "#0b0b0b"
            ax.text(j, i, matriz[i, j], ha="center", va="center", fontsize=7, color=cor_texto)
    ax.set_xticks(range(n), nomes, rotation=45, ha="right")
    ax.set_yticks(range(n), nomes)
    ax.set(title="Matriz de confusão (teste)", xlabel="prevista", ylabel="verdadeira")
    fig.tight_layout()
    plt.show()


def plot_regressao_real_vs_previsto(real, previsto, titulo="Teste: real × previsto"):
    """Cada ponto é uma casa; quanto mais perto da diagonal, melhor."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(real, previsto, s=8, alpha=0.4, color=COR_TREINO)
    limites = [min(real.min(), previsto.min()), max(real.max(), previsto.max())]
    ax.plot(limites, limites, color="#8a8984", linewidth=1, linestyle="--")
    ax.set(title=titulo, xlabel="valor real", ylabel="valor previsto")
    fig.tight_layout()
    plt.show()


def imprimir_tabela(resultados):
    """resultados: {nome_do_modelo: {nome_metrica: valor}} -> tabela em texto."""
    primeiro_modelo = list(resultados.values())[0]
    nomes_metricas = list(primeiro_modelo.keys())
    largura = max(len(nome) for nome in resultados) + 2

    cabecalho = "modelo".ljust(largura)
    for metrica in nomes_metricas:
        cabecalho += metrica.rjust(12)
    print(cabecalho)
    print("-" * len(cabecalho))
    for nome, metricas in resultados.items():
        linha = nome.ljust(largura)
        for metrica in nomes_metricas:
            linha += f"{metricas[metrica]:12.4f}"
        print(linha)
