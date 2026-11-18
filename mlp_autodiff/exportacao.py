"""
exportacao.py — Salvar e carregar o modelo em um arquivo JSON legível.

O arquivo contém TUDO o que é preciso para inferir sem treinar:
  * a topologia (config: camadas, ativações, inicialização...)
  * os pesos (W e b de cada camada, como listas)
  * as estatísticas de padronização (média/desvio de X e, na regressão, de y).
    Sem elas, a inferência depois de importar daria resultado errado!
  * nome da loss e métricas finais (só informativo)
"""
import json

import numpy as np

from rede import MLP

VERSAO_FORMATO = 1


def _para_lista(valor):
    """Converte arrays NumPy em listas (JSON não entende ndarray)."""
    if valor is None:
        return None
    return np.asarray(valor, dtype=np.float64).tolist()


def salvar_modelo(rede, caminho, nome_perda="", metricas=None):
    """Grava config + pesos + padronização em `caminho` (.json)."""
    camadas = []
    for camada in rede.camadas:
        camadas.append({"W": camada.W.data.tolist(), "b": camada.b.data.tolist()})

    padronizacao = None
    if rede.padronizacao is not None:
        padronizacao = {}
        for chave, valor in rede.padronizacao.items():
            padronizacao[chave] = _para_lista(valor)

    conteudo = {
        "versao": VERSAO_FORMATO,
        "config": rede.config,
        "perda": nome_perda,
        "camadas": camadas,
        "padronizacao": padronizacao,
        "metricas": metricas or {},
    }
    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(conteudo, arquivo, indent=1)


def carregar_modelo(caminho):
    """Reconstrói o MLP a partir do arquivo: pronto para `rede.prever(X)`."""
    with open(caminho, "r", encoding="utf-8") as arquivo:
        conteudo = json.load(arquivo)

    rede = MLP(conteudo["config"])  # cria a topologia (pesos aleatórios)
    for camada, salvos in zip(rede.camadas, conteudo["camadas"]):
        camada.W.data = np.array(salvos["W"], dtype=np.float64)  # sobrescreve
        camada.b.data = np.array(salvos["b"], dtype=np.float64)
    rede.padronizacao = conteudo["padronizacao"]
    return rede
