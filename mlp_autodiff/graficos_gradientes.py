"""
graficos_gradientes.py — Gráficos de monitoramento interno da rede:
histogramas de ativações/gradientes, norma do gradiente por camada,
comparações em função da profundidade e % de neurônios ReLU mortos.

Cores: camadas vão do azul claro (1ª camada) ao escuro (última).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from graficos import CORES, COR_TREINO, COR_VAL, cor_da_camada


def plot_histogramas(historico, tipo="ativacoes", max_linhas=4):
    """Grade de histogramas: linhas = épocas, colunas = camadas.

    tipo: "ativacoes", "grad_ativacoes" ou "grad_pesos".
    Gradientes são mostrados como log10(|g|): assim dá para ver se eles
    ficam 1000x menores (dissipação) ou maiores (explosão) de uma camada para outra.
    """
    registros = historico["monitoramento"]
    escolhidos = np.unique(np.linspace(0, len(registros) - 1, max_linhas).astype(int))
    n_camadas = len(registros[0][tipo])
    # Gradientes: mesmo eixo x na linha, para comparar ordens de grandeza entre camadas.
    compartilhar_x = "row" if tipo != "ativacoes" else False
    fig, eixos = plt.subplots(len(escolhidos), n_camadas, squeeze=False, sharex=compartilhar_x,
                              figsize=(2.1 * n_camadas, 1.7 * len(escolhidos)))
    for linha, indice_registro in enumerate(escolhidos):
        registro = registros[indice_registro]
        for camada in range(n_camadas):
            valores = registro[tipo][camada].ravel()
            if tipo != "ativacoes":
                valores = np.log10(np.abs(valores) + 1e-30)
            valores = valores[np.isfinite(valores)]
            ax = eixos[linha, camada]
            ax.hist(valores, bins=40, color=cor_da_camada(camada, n_camadas))
            ax.set_yticks([])
            if linha == 0:
                ax.set_title(f"camada {camada + 1}")
            if camada == 0:
                ax.set_ylabel(f"época {registro['epoca']}")
    rotulos = {"ativacoes": "valor da ativação", "grad_ativacoes": "log10 |dL/d ativação|",
               "grad_pesos": "log10 |dL/dW|"}
    fig.suptitle(f"Histogramas — {rotulos[tipo]}")
    fig.tight_layout()
    plt.show()


def _iteracoes_para_plotar(historico, max_iteracoes):
    """Quantas iterações mostrar: até `max_iteracoes`, ou até o custo virar inf/NaN."""
    perdas = np.array(historico["perda_passos"][:max_iteracoes])
    nao_finitos = np.where(~np.isfinite(perdas))[0]
    if len(nao_finitos) > 0:
        return nao_finitos[0] + 1  # inclui a iteração em que divergiu
    return len(perdas)


def _desenhar_normas(ax, historico, titulo, por_iteracao=False, max_iteracoes=100):
    """Uma linha por camada. Por época (média) ou por iteração (cada mini-lote)."""
    if por_iteracao:
        n = _iteracoes_para_plotar(historico, max_iteracoes)
        normas = np.array(historico["norma_grad_passos"][:n])
        rotulo_x = "iteração"
    else:
        normas = np.array(historico["norma_grad"])  # (épocas, camadas)
        rotulo_x = "época"
    eixo_x = np.arange(1, len(normas) + 1)
    n_camadas = normas.shape[1]
    marcador = "o" if len(normas) < 10 else None  # poucos pontos: mostra cada um
    for camada in range(n_camadas):
        ax.plot(eixo_x, normas[:, camada], color=cor_da_camada(camada, n_camadas),
                marker=marcador, markersize=5, label=f"camada {camada + 1}")
    ax.set_yscale("log")
    ax.set_xlim(0.5, len(normas) + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(title=titulo, xlabel=rotulo_x, ylabel="‖dL/dW‖ (escala log)")


def _desenhar_custo(ax, historico, por_iteracao=False, max_iteracoes=100):
    """Custo por época (treino e validação) ou por iteração (mini-lote)."""
    if por_iteracao:
        n = _iteracoes_para_plotar(historico, max_iteracoes)
        perdas = np.array(historico["perda_passos"][:n])
        marcador = "o" if n < 10 else None
        ax.plot(np.arange(1, n + 1), perdas, color=COR_TREINO,
                marker=marcador, markersize=5, label="treino (mini-lote)")
        if not np.isfinite(perdas[-1]):  # marca a iteração em que o custo virou inf/NaN
            ax.axvline(n, color="#e34948", linestyle="--", linewidth=1, label=f"custo = {perdas[-1]}")
        ax.set_xlim(0.5, n + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel="iteração", ylabel="custo (escala log)", yscale="log")
        return
    epocas = np.arange(1, len(historico["perda_treino"]) + 1)
    ax.plot(epocas, historico["perda_treino"], color=COR_TREINO, label="treino")
    ax.plot(epocas, historico["perda_val"], color=COR_VAL, label="validação")
    ax.set(xlabel="época", ylabel="custo (escala log)", yscale="log")


def plot_norma_grad_por_camada(historico, titulo="Norma do gradiente por camada", por_iteracao=False):
    """Uma linha por camada ao longo do treino (eixo y em log)."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _desenhar_normas(ax, historico, titulo, por_iteracao)
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    plt.show()


def plot_antes_depois(hist_antes, hist_depois, titulo_antes, titulo_depois, por_iteracao=False):
    """Lado a lado: norma por camada (em cima) e custo (embaixo)."""
    fig, eixos = plt.subplots(2, 2, figsize=(12, 7))
    pares = [(hist_antes, titulo_antes), (hist_depois, titulo_depois)]
    for coluna, (hist, titulo) in enumerate(pares):
        _desenhar_normas(eixos[0, coluna], hist, titulo, por_iteracao)
        _desenhar_custo(eixos[1, coluna], hist, por_iteracao)
    eixos[0, 1].legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    eixos[1, 0].legend()
    fig.tight_layout()
    plt.show()


def _plot_por_profundidade(historicos, valores_por_camada, titulo, ylabel):
    """Uma linha por configuração; eixo x = camada; eixo y em log."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (nome, historico) in enumerate(historicos.items()):
        valores = valores_por_camada(historico)
        camadas = np.arange(1, len(valores) + 1)
        ax.plot(camadas, valores, color=CORES[i], marker="o", markersize=5, label=nome)
    ax.set_yscale("log")
    ax.set(title=titulo, xlabel="camada (1 = mais perto da entrada)", ylabel=ylabel)
    ax.legend()
    fig.tight_layout()
    plt.show()


def _norma_na_primeira_iteracao(historico):
    return historico["norma_grad_passos"][0]


def _media_abs_ativacao_inicial(historico):
    return [np.mean(np.abs(a)) for a in historico["monitoramento"][0]["ativacoes"]]


def plot_norma_vs_profundidade(historicos):
    """‖dL/dW‖ em função da camada na 1ª iteração (rede recém-inicializada)."""
    _plot_por_profundidade(historicos, _norma_na_primeira_iteracao,
                           "Norma do gradiente vs. profundidade (1ª iteração)", "‖dL/dW‖ (escala log)")


def plot_ativacao_vs_profundidade(historicos):
    """Média de |ativação| por camada, na rede recém-inicializada (época 0)."""
    _plot_por_profundidade(historicos, _media_abs_ativacao_inicial,
                           "Tamanho do sinal no forward vs. profundidade (época 0)",
                           "média |ativação| (escala log)")


def porcentagem_relu_mortos(historico):
    """% de unidades com ativação <= 0 em TODO o lote fixo, por camada oculta."""
    tabela = []
    for registro in historico["monitoramento"]:
        linha = []
        for ativacao in registro["ativacoes"][:-1]:  # a última camada é a saída
            morta = np.all(ativacao <= 0, axis=0)    # unidade zerada em todos os exemplos
            linha.append(100 * np.mean(morta))
        tabela.append(linha)
    return np.array(tabela)  # (registros, camadas_ocultas)


def plot_relu_mortos(historico, titulo="Neurônios ReLU mortos"):
    tabela = porcentagem_relu_mortos(historico)
    epocas = [r["epoca"] for r in historico["monitoramento"]]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for camada in range(tabela.shape[1]):
        ax.plot(epocas, tabela[:, camada], color=cor_da_camada(camada, tabela.shape[1]),
                marker="o", markersize=5, label=f"camada {camada + 1}")
    ax.set(title=titulo, xlabel="época", ylabel="% de unidades mortas", ylim=(-2, 102))
    ax.legend(fontsize=7, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    plt.show()
