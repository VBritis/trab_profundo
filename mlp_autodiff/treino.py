"""
treino.py — Loop de treino com mini-lotes, métricas, histórico,
coleta de dados para os histogramas e early stopping.
"""
import numpy as np

from motor import Tensor
from rede import PERDAS
from otimizadores import OTIMIZADORES

# Valores usados quando a chave não aparece no config_treino.
CONFIG_TREINO_PADRAO = {
    "perda": "mse",              # "mse", "bce" ou "entropia_cruzada"
    "metrica": "r2",             # "r2" ou "acuracia"
    "otimizador": "adam",        # "sgd" ou "adam"
    "taxa_aprendizado": 1e-3,
    "epocas": 20,
    "tamanho_lote": 64,
    "clip_norma": None,          # ex.: 1.0 liga o gradient clipping
    "lambda_l2": 0.0,            # força da regularização L2 (0 = desligada)
    "paciencia": None,           # early stopping (None = desligado)
    "coletar_a_cada": 5,         # de quantas em quantas épocas coletar histogramas
    "tamanho_lote_fixo": 256,    # lote FIXO de validação para os histogramas
    "seed": 0,
    "verbose": True,
}


# ----------------------------------------------------------------------
# Métricas (NumPy puro: não precisam de gradiente)
# ----------------------------------------------------------------------
def r2(predicao, alvo):
    """R² = 1 - SS_res / SS_tot. 1 = perfeito; 0 = igual a prever a média."""
    ss_res = np.sum((alvo - predicao) ** 2)
    ss_tot = np.sum((alvo - alvo.mean(axis=0)) ** 2)
    return 1.0 - ss_res / ss_tot


def acuracia(predicao, alvo):
    """Fração de acertos. 1 saída: limiar 0.5 ; várias saídas: argmax."""
    if alvo.shape[1] == 1:
        return np.mean((predicao > 0.5) == (alvo > 0.5))
    return np.mean(np.argmax(predicao, axis=1) == np.argmax(alvo, axis=1))


METRICAS = {"r2": r2, "acuracia": acuracia}


def metricas_regressao(real, previsto):
    """MSE, R² e MAE (use com valores na escala ORIGINAL do alvo)."""
    return {"MSE": float(np.mean((real - previsto) ** 2)),
            "R2": float(r2(previsto, real)),
            "MAE": float(np.mean(np.abs(real - previsto)))}


# ----------------------------------------------------------------------
# Perda, avaliação e monitoramento
# ----------------------------------------------------------------------
def penalidade_l2(rede):
    """Σ W² de todas as camadas. Nenhuma derivada foi escrita aqui."""
    total = Tensor(0.0)
    for camada in rede.camadas:
        total = total + (camada.W ** 2).sum()
    return total


def calcular_perda(rede, X, Y, nome_perda, lambda_l2=0.0):
    """Forward + loss. Devolve (perda: Tensor escalar, predição: Tensor)."""
    predicao = rede.forward(X)
    perda = PERDAS[nome_perda](predicao, Tensor(Y))
    if lambda_l2 > 0:
        perda = perda + lambda_l2 * penalidade_l2(rede)
    return perda, predicao


def avaliar(rede, X, Y, nome_perda, nome_metrica):
    """Perda e métrica em um conjunto inteiro (sem treinar)."""
    perda, predicao = calcular_perda(rede, X, Y, nome_perda)
    return float(perda.data), float(METRICAS[nome_metrica](predicao.data, Y))


def normas_por_camada(rede):
    """‖dL/dW‖ de cada camada (para o gráfico de dissipação/explosão)."""
    normas = []
    for camada in rede.camadas:
        normas.append(np.linalg.norm(camada.W.grad))
    return normas


def coletar_monitoramento(rede, X_fixo, Y_fixo, nome_perda, epoca):
    """Forward + backward no lote FIXO e guarda cópias para os histogramas."""
    for p in rede.parametros():
        p.grad = np.zeros_like(p.data)
    perda, _ = calcular_perda(rede, X_fixo, Y_fixo, nome_perda)
    perda.backward()
    registro = {
        "epoca": epoca,
        "ativacoes": [a.data.copy() for a in rede.ativacoes],
        "grad_ativacoes": [a.grad.copy() for a in rede.ativacoes],
        "grad_pesos": [c.W.grad.copy() for c in rede.camadas],
    }
    for p in rede.parametros():  # não deixa "sujeira" para o treino
        p.grad = np.zeros_like(p.data)
    return registro


def gerar_minilotes(X, Y, tamanho_lote, rng):
    """Embaralha os índices e corta em pedaços de `tamanho_lote`."""
    indices = rng.permutation(len(X))
    lotes = []
    for inicio in range(0, len(X), tamanho_lote):
        selecionados = indices[inicio:inicio + tamanho_lote]
        lotes.append((X[selecionados], Y[selecionados]))
    return lotes


def copiar_pesos(rede):
    return [p.data.copy() for p in rede.parametros()]


def restaurar_pesos(rede, pesos):
    for p, valor in zip(rede.parametros(), pesos):
        p.data = valor.copy()


# ----------------------------------------------------------------------
# Loop principal
# ----------------------------------------------------------------------
def treinar(rede, dados, config_treino):
    """Treina `rede` com dados["X_treino"], ... e devolve o histórico."""
    cfg = dict(CONFIG_TREINO_PADRAO)
    cfg.update(config_treino)
    rng = np.random.default_rng(cfg["seed"])
    classe_otimizador = OTIMIZADORES[cfg["otimizador"]]
    otimizador = classe_otimizador(rede.parametros(), cfg["taxa_aprendizado"], cfg["clip_norma"])

    X_fixo = dados["X_val"][:cfg["tamanho_lote_fixo"]]
    Y_fixo = dados["Y_val"][:cfg["tamanho_lote_fixo"]]
    historico = {"perda_treino": [], "perda_val": [], "metrica_treino": [],
                 "metrica_val": [], "norma_grad": [], "monitoramento": [],
                 "norma_grad_passos": [], "perda_passos": [],  # por iteração
                 "metrica": cfg["metrica"], "melhor_epoca": None}
    historico["monitoramento"].append(coletar_monitoramento(rede, X_fixo, Y_fixo, cfg["perda"], 0))
    melhor_perda_val = np.inf
    melhores_pesos = None
    epocas_sem_melhora = 0

    for epoca in range(1, cfg["epocas"] + 1):
        normas_da_epoca = []
        for X_lote, Y_lote in gerar_minilotes(dados["X_treino"], dados["Y_treino"], cfg["tamanho_lote"], rng):
            otimizador.zerar_grad()  # 1) zera (gradientes acumulam!)
            perda, _ = calcular_perda(rede, X_lote, Y_lote, cfg["perda"], cfg["lambda_l2"])
            perda.backward()         # 2) gradientes pelo motor
            normas_da_epoca.append(normas_por_camada(rede))
            historico["perda_passos"].append(float(perda.data))
            otimizador.passo()       # 3) atualiza os pesos
        historico["norma_grad"].append(np.mean(normas_da_epoca, axis=0))
        historico["norma_grad_passos"].extend(normas_da_epoca)

        perda_tr, metrica_tr = avaliar(rede, dados["X_treino"], dados["Y_treino"], cfg["perda"], cfg["metrica"])
        perda_val, metrica_val = avaliar(rede, dados["X_val"], dados["Y_val"], cfg["perda"], cfg["metrica"])
        historico["perda_treino"].append(perda_tr)
        historico["perda_val"].append(perda_val)
        historico["metrica_treino"].append(metrica_tr)
        historico["metrica_val"].append(metrica_val)
        if cfg["verbose"]:
            print(f"época {epoca:3d} | perda treino {perda_tr:.4f} val {perda_val:.4f} "
                  f"| {cfg['metrica']} treino {metrica_tr:.4f} val {metrica_val:.4f}")

        if not np.isfinite(perda_tr):
            print(f"Treino DIVERGIU na época {epoca} (perda = {perda_tr}). Parando.")
            break
        if epoca % cfg["coletar_a_cada"] == 0:
            historico["monitoramento"].append(coletar_monitoramento(rede, X_fixo, Y_fixo, cfg["perda"], epoca))

        # Early stopping: guarda os pesos da melhor época na validação.
        if cfg["paciencia"] is not None:
            if perda_val < melhor_perda_val:
                melhor_perda_val = perda_val
                melhores_pesos = copiar_pesos(rede)
                historico["melhor_epoca"] = epoca
                epocas_sem_melhora = 0
            else:
                epocas_sem_melhora += 1
            if epocas_sem_melhora >= cfg["paciencia"]:
                print(f"Early stopping na época {epoca} (melhor: {historico['melhor_epoca']}).")
                break

    if melhores_pesos is not None:
        restaurar_pesos(rede, melhores_pesos)
    return historico
