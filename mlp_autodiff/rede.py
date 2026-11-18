"""
rede.py — Inicializações, ativações, funções de custo, Camada e MLP.

IMPORTANTE: nenhuma derivada é escrita neste arquivo. Tudo é composto
pelas operações primitivas do motor; o backward sai "de graça".
"""
import numpy as np

from motor import Tensor, _como_tensor


# ======================================================================
# Inicializações dos pesos (o bias é sempre zero)
# Todas devolvem uma matriz (fan_in, fan_out).
# fan_in = nº de entradas da camada ; fan_out = nº de saídas.
# ======================================================================
def init_he(fan_in, fan_out, rng, escala):
    # N(0, √(2/fan_in)). A ReLU zera metade das entradas, o que corta a
    # variância pela metade; o "2" compensa isso e mantém a variância estável.
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))


def init_xavier(fan_in, fan_out, rng, escala):
    # N(0, √(2/(fan_in+fan_out))). Equilibra a variância no forward (fan_in)
    # e no backward (fan_out). Pensada para ativações simétricas: Tanh/Sigmoid.
    desvio = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, desvio, size=(fan_in, fan_out))


def init_normal(fan_in, fan_out, rng, escala):
    # N(0, escala). Ignora o tamanho da camada: com `escala` pequena o sinal
    # encolhe a cada camada (dissipação); grande, ele cresce (explosão).
    return rng.normal(0.0, escala, size=(fan_in, fan_out))


def init_uniforme(fan_in, fan_out, rng, escala):
    # U(-escala, +escala).
    return rng.uniform(-escala, escala, size=(fan_in, fan_out))


INICIALIZACOES = {
    "he": init_he,
    "xavier": init_xavier,
    "normal": init_normal,
    "uniforme": init_uniforme,
}


# ======================================================================
# Ativações (recebem e devolvem Tensor)
# ======================================================================
def sigmoid(x):
    return x.sigmoid()  # primitiva do motor (versão estável)


def tanh(x):
    return x.tanh()  # primitiva do motor


def relu(x):
    # relu(x) = x * mascara, com mascara = 1 onde x > 0 e 0 onde x <= 0.
    # A máscara é uma CONSTANTE (Tensor sem pais). Então o motor deriva
    # sozinho: d(x*m)/dx = m. Não precisamos de backward próprio.
    mascara = Tensor((x.data > 0) * 1.0)
    return x * mascara


def leaky_relu(x, alfa=0.01):
    # Mesma ideia da ReLU, mas a máscara vale `alfa` (e não 0) onde x <= 0.
    mascara = Tensor(np.where(x.data > 0, 1.0, alfa))
    return x * mascara


def linear(x):
    return x


def softmax(x):
    # softmax(x)_k = e^(x_k) / Σ_j e^(x_j), linha a linha.
    # Subtrair o máximo da linha não muda o resultado (cancela na divisão),
    # mas evita e^(número grande) = inf. O máximo entra como CONSTANTE.
    # Nenhuma derivada foi escrita aqui: só exp, soma e divisão do motor.
    maximo = Tensor(x.data.max(axis=1, keepdims=True))
    exponenciais = (x - maximo).exp()
    return exponenciais / exponenciais.sum(axis=1, keepdims=True)


ATIVACOES = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "linear": linear,
    "softmax": softmax,
}


# ======================================================================
# Funções de custo (recebem predição e alvo; devolvem Tensor escalar)
# ======================================================================
def mse(predicao, alvo):
    # média de (pred - alvo)²
    return ((predicao - alvo) ** 2).mean()


def bce(predicao, alvo, eps=1e-7):
    # -média[ y·log(p) + (1-y)·log(1-p) ]
    # eps evita log(0) = -inf quando p chega a 0 ou 1.
    # Multiclasse: alvo em one-hot e UMA sigmoid por saída; cada saída
    # responde sozinha "é da classe k?" (k problemas binários em paralelo).
    termo_positivo = alvo * (predicao + eps).log()
    termo_negativo = (1.0 - alvo) * (1.0 - predicao + eps).log()
    return -(termo_positivo + termo_negativo).mean()


def entropia_cruzada(probabilidades, alvo, eps=1e-7):
    # -média_exemplos[ Σ_k y_k·log(p_k) ]  (usar com saída softmax)
    # Nenhuma derivada foi escrita aqui.
    por_exemplo = (alvo * (probabilidades + eps).log()).sum(axis=1)
    return -por_exemplo.mean()


PERDAS = {
    "mse": mse,
    "bce": bce,
    "entropia_cruzada": entropia_cruzada,
}


# ======================================================================
# Camada e MLP
# ======================================================================
class Camada:
    """Camada densa: saida = ativacao(x @ W + b)."""

    def __init__(self, n_entrada, n_saida, ativacao, inicializacao, rng, escala, indice):
        pesos = INICIALIZACOES[inicializacao](n_entrada, n_saida, rng, escala)
        self.W = Tensor(pesos, nome=f"W{indice}")               # (n_entrada, n_saida)
        self.b = Tensor(np.zeros((1, n_saida)), nome=f"b{indice}")  # (1, n_saida)
        self.ativacao = ativacao  # string, ex.: "relu"

    def forward(self, x):
        # x: (batch, n_entrada) @ W: (n_entrada, n_saida) -> (batch, n_saida)
        # + b: (1, n_saida) é "esticado" para todas as linhas (broadcasting).
        z = x @ self.W + self.b
        return ATIVACOES[self.ativacao](z)


class MLP:
    """Rede de várias camadas densas, configurada por um dicionário simples."""

    def __init__(self, config):
        self.config = config
        rng = np.random.default_rng(config.get("seed", 0))
        escala = config.get("escala_init", 0.1)  # usada por "normal" e "uniforme"
        tamanhos = [config["n_entrada"]] + list(config["camadas_ocultas"]) + [config["n_saida"]]

        self.camadas = []
        n_camadas = len(tamanhos) - 1
        for i in range(n_camadas):
            if i == n_camadas - 1:
                ativacao = config["ativacao_saida"]
            else:
                ativacao = config["ativacao_oculta"]
            camada = Camada(tamanhos[i], tamanhos[i + 1], ativacao,
                            config["inicializacao"], rng, escala, indice=i + 1)
            self.camadas.append(camada)

        self.ativacoes = []        # saída de cada camada no último forward
        self.padronizacao = None   # médias/desvios dos dados (vai no arquivo exportado)

    def forward(self, x):
        """x: dados JÁ padronizados (array ou Tensor). Devolve Tensor."""
        x = _como_tensor(x)
        # Guardamos a saída de cada camada para os histogramas. Depois do
        # backward, `.grad` desses tensores é dL/d(ativação): vem de graça
        # do autodiff, sem nenhum código extra.
        self.ativacoes = []
        for camada in self.camadas:
            x = camada.forward(x)
            self.ativacoes.append(x)
        return x

    def parametros(self):
        """Lista de todos os Tensores treináveis: [W1, b1, W2, b2, ...]."""
        lista = []
        for camada in self.camadas:
            lista.append(camada.W)
            lista.append(camada.b)
        return lista

    def prever(self, X_bruto):
        """Inferência a partir de dados na escala ORIGINAL (array NumPy).

        Aplica a padronização da entrada, roda a rede e, na regressão,
        desfaz a padronização do alvo. Devolve array NumPy.
        """
        X = np.asarray(X_bruto, dtype=np.float64)
        p = self.padronizacao
        if p is not None:
            X = (X - np.asarray(p["media_x"])) / np.asarray(p["desvio_x"])
        saida = self.forward(X).data
        if p is not None and p.get("media_y") is not None:
            saida = saida * np.asarray(p["desvio_y"]) + np.asarray(p["media_y"])
        return saida
