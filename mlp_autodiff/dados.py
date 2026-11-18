"""
dados.py — Carregar, dividir (treino/val/teste) e padronizar os datasets.

scikit-learn é usado SOMENTE para baixar os dados e fazer o split.

Cada carregador devolve um dicionário com:
  X_treino, Y_treino, X_val, Y_val, X_teste, Y_teste  -> prontos para a rede
  X_teste_bruto, Y_teste_bruto                        -> escala original
  padronizacao                                        -> vai no arquivo exportado
"""
import os

import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split

PASTA_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados_cache")
NOMES_FASHION = ["Camiseta", "Calça", "Pulôver", "Vestido", "Casaco",
                 "Sandália", "Camisa", "Tênis", "Bolsa", "Bota"]


def dividir_70_15_15(X, Y, seed):
    """Split fixo: 70% treino, 15% validação, 15% teste."""
    X_treino, X_resto, Y_treino, Y_resto = train_test_split(X, Y, test_size=0.30, random_state=seed)
    X_val, X_teste, Y_val, Y_teste = train_test_split(X_resto, Y_resto, test_size=0.50, random_state=seed)
    return X_treino, Y_treino, X_val, Y_val, X_teste, Y_teste


def para_one_hot(rotulos, n_classes):
    """[2, 0] -> [[0,0,1,...], [1,0,0,...]]"""
    return np.eye(n_classes)[rotulos]


def carregar_california(modo_rapido=True, seed=42):
    """Regressão: preço médio das casas por quarteirão na Califórnia (8 atributos).

    X e y são padronizados (média 0, desvio 1) usando SOMENTE o treino:
    usar val/teste no cálculo da média seria "espiar" dados futuros.
    """
    conjunto = fetch_california_housing(data_home=PASTA_CACHE)
    X = conjunto.data.astype(np.float64)
    Y = conjunto.target.astype(np.float64).reshape(-1, 1)  # (n,) -> (n, 1)
    if modo_rapido:
        sorteio = np.random.default_rng(seed).choice(len(X), size=6000, replace=False)
        X, Y = X[sorteio], Y[sorteio]

    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = dividir_70_15_15(X, Y, seed)
    padronizacao = {"media_x": X_tr.mean(axis=0), "desvio_x": X_tr.std(axis=0),
                    "media_y": Y_tr.mean(axis=0), "desvio_y": Y_tr.std(axis=0)}

    def padronizar_x(M):
        return (M - padronizacao["media_x"]) / padronizacao["desvio_x"]

    def padronizar_y(M):
        return (M - padronizacao["media_y"]) / padronizacao["desvio_y"]

    return {"X_treino": padronizar_x(X_tr), "Y_treino": padronizar_y(Y_tr),
            "X_val": padronizar_x(X_val), "Y_val": padronizar_y(Y_val),
            "X_teste": padronizar_x(X_te), "Y_teste": padronizar_y(Y_te),
            "X_teste_bruto": X_te, "Y_teste_bruto": Y_te,
            "padronizacao": padronizacao, "nomes_atributos": list(conjunto.feature_names)}


def _baixar_fashion_mnist():
    """Baixa do OpenML uma única vez e guarda em .npz (ler o .npz é muito
    mais rápido do que reprocessar o arquivo original a cada execução)."""
    caminho = os.path.join(PASTA_CACHE, "fashion_mnist.npz")
    if not os.path.exists(caminho):
        os.makedirs(PASTA_CACHE, exist_ok=True)
        # parser="liac-arff": o parser via pandas falha com versões novas do pandas.
        conjunto = fetch_openml("Fashion-MNIST", version=1, as_frame=False,
                                data_home=PASTA_CACHE, parser="liac-arff")
        np.savez_compressed(caminho, X=conjunto.data.astype(np.uint8),
                            y=conjunto.target.astype(np.int64))
    arquivo = np.load(caminho)
    return arquivo["X"], arquivo["y"]


def carregar_fashion_mnist(modo_rapido=True, seed=42):
    """Classificação: imagens 28x28 de roupas, 10 classes.

    Entrada: pixels / 255 (fica em [0, 1]). Isso também é uma padronização
    (média 0, desvio 255) e por isso vai para o arquivo exportado.
    Alvo: one-hot com 10 colunas.
    """
    X, rotulos = _baixar_fashion_mnist()
    X = X.astype(np.float64)
    if modo_rapido:
        sorteio = np.random.default_rng(seed).choice(len(X), size=8000, replace=False)
        X, rotulos = X[sorteio], rotulos[sorteio]
    Y = para_one_hot(rotulos, 10)

    X_tr, Y_tr, X_val, Y_val, X_te, Y_te = dividir_70_15_15(X, Y, seed)
    padronizacao = {"media_x": 0.0, "desvio_x": 255.0, "media_y": None, "desvio_y": None}
    return {"X_treino": X_tr / 255.0, "Y_treino": Y_tr,
            "X_val": X_val / 255.0, "Y_val": Y_val,
            "X_teste": X_te / 255.0, "Y_teste": Y_te,
            "X_teste_bruto": X_te, "Y_teste_bruto": Y_te,
            "padronizacao": padronizacao, "nomes_classes": NOMES_FASHION}
