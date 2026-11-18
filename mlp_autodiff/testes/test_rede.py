"""
test_rede.py — Provas de corretude ponta a ponta do MLP.

1) Gradient check da rede inteira (cada ativação, cada loss).
2) Equivalência com regressão linear (solução fechada por mínimos quadrados).
3) Overfit de um mini-lote de 16 exemplos.
4) Round-trip de exportação: salvar -> carregar -> predições idênticas.
"""
import numpy as np
import pytest

from motor import Tensor
from rede import MLP, PERDAS
from checagem import verificar_gradientes
from treino import treinar, calcular_perda
from exportacao import salvar_modelo, carregar_modelo

TOLERANCIA = 1e-6


def config_pequena(ativacao_oculta, ativacao_saida, n_saida=2):
    return {"n_entrada": 3, "camadas_ocultas": [5, 4], "n_saida": n_saida,
            "ativacao_oculta": ativacao_oculta, "ativacao_saida": ativacao_saida,
            "inicializacao": "xavier", "seed": 1}


@pytest.mark.parametrize("ativacao", ["sigmoid", "tanh", "relu", "leaky_relu", "linear"])
@pytest.mark.parametrize("ativacao_saida, nome_perda", [("linear", "mse"),
                                                        ("sigmoid", "bce"),
                                                        ("softmax", "entropia_cruzada")])
def test_gradiente_da_rede_inteira(ativacao, ativacao_saida, nome_perda):
    # ReLU/Leaky ReLU não são deriváveis em 0: se uma pré-ativação cair a
    # menos de h=1e-6 de 0, a diferença finita "pula a quina" e erra.
    # Com entradas contínuas aleatórias, a chance disso é desprezível.
    rng = np.random.default_rng(7)
    rede = MLP(config_pequena(ativacao, ativacao_saida))
    X = rng.normal(size=(6, 3))
    Y = np.eye(2)[rng.integers(0, 2, size=6)]  # one-hot (serve para as 3 losses)

    def funcao():
        perda, _ = calcular_perda(rede, X, Y, nome_perda, lambda_l2=0.01)
        return perda
    assert verificar_gradientes(funcao, rede.parametros()) < TOLERANCIA


def test_equivalencia_com_regressao_linear():
    # MLP sem camadas ocultas + saída linear + MSE = regressão linear.
    # O mínimo da MSE tem solução fechada (mínimos quadrados): os pesos
    # treinados pelo motor + Adam têm que chegar nela.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([[2.0], [-1.0], [0.5]]) + 0.7 + 0.1 * rng.normal(size=(200, 1))

    config = {"n_entrada": 3, "camadas_ocultas": [], "n_saida": 1,
              "ativacao_oculta": "linear", "ativacao_saida": "linear",
              "inicializacao": "normal", "seed": 0}
    rede = MLP(config)
    dados = {"X_treino": X, "Y_treino": y, "X_val": X, "Y_val": y}
    treinar(rede, dados, {"perda": "mse", "otimizador": "adam", "taxa_aprendizado": 0.05,
                          "epocas": 1500, "tamanho_lote": 200, "verbose": False,
                          "coletar_a_cada": 10_000})

    X_com_um = np.hstack([X, np.ones((200, 1))])  # coluna de 1 = bias
    solucao = np.linalg.lstsq(X_com_um, y, rcond=None)[0]  # [0] = os coeficientes
    pesos_rede = np.vstack([rede.camadas[0].W.data, rede.camadas[0].b.data])
    assert np.allclose(pesos_rede, solucao, atol=1e-4)


@pytest.mark.parametrize("ativacao", ["tanh", "relu"])
def test_overfit_de_um_mini_lote(ativacao):
    # 16 exemplos e uma rede com muitos parâmetros: ela tem que DECORAR.
    # Se a perda não vai a ~0, há bug no motor, na rede ou no otimizador.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(16, 4))
    y = rng.normal(size=(16, 1))
    config = {"n_entrada": 4, "camadas_ocultas": [32, 32], "n_saida": 1,
              "ativacao_oculta": ativacao, "ativacao_saida": "linear",
              "inicializacao": "he", "seed": 0}
    rede = MLP(config)
    dados = {"X_treino": X, "Y_treino": y, "X_val": X, "Y_val": y}
    historico = treinar(rede, dados, {"perda": "mse", "taxa_aprendizado": 0.01, "epocas": 600,
                                      "tamanho_lote": 16, "verbose": False,
                                      "coletar_a_cada": 10_000})
    assert historico["perda_treino"][-1] < 1e-3


def test_round_trip_de_exportacao(tmp_path):
    rng = np.random.default_rng(9)
    rede = MLP(config_pequena("relu", "linear", n_saida=1))
    rede.padronizacao = {"media_x": rng.normal(size=3), "desvio_x": rng.uniform(1, 2, size=3),
                         "media_y": [5.0], "desvio_y": [2.0]}
    X_bruto = rng.normal(size=(10, 3))

    caminho = tmp_path / "modelo.json"
    salvar_modelo(rede, caminho, nome_perda="mse")
    carregada = carregar_modelo(caminho)

    assert np.allclose(rede.prever(X_bruto), carregada.prever(X_bruto))
    assert carregada.config == rede.config


def test_padronizacao_e_aplicada_na_inferencia():
    # prever() com dados brutos == forward() com dados padronizados à mão.
    rede = MLP(config_pequena("tanh", "linear", n_saida=1))
    rede.padronizacao = {"media_x": [1.0, 2.0, 3.0], "desvio_x": [2.0, 2.0, 2.0],
                         "media_y": [10.0], "desvio_y": [3.0]}
    X_bruto = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    esperado = rede.forward((X_bruto - [1.0, 2.0, 3.0]) / 2.0).data * 3.0 + 10.0
    assert np.allclose(rede.prever(X_bruto), esperado)


def test_bce_com_predicao_extrema_nao_da_nan():
    # p = 0 e p = 1 exatos: sem o eps, log(0) = -inf.
    p = Tensor([[0.0, 1.0]])
    perda = PERDAS["bce"](p, Tensor([[1.0, 0.0]]))
    perda.backward()
    assert np.isfinite(perda.data) and np.all(np.isfinite(p.grad))
