"""
otimizadores.py — SGD e Adam, com gradient clipping por norma opcional.

Uso no loop de treino:
    otimizador.zerar_grad()   # 1) limpa gradientes do passo anterior
    perda.backward()          # 2) calcula novos gradientes
    otimizador.passo()        # 3) atualiza os pesos
"""
import numpy as np


def norma_global(parametros):
    """√( Σ de todos os gradientes ao quadrado ), juntando todos os parâmetros."""
    total = 0.0
    for p in parametros:
        total += np.sum(p.grad ** 2)
    return np.sqrt(total)


def recortar_gradientes(parametros, norma_maxima):
    """Gradient clipping: se a norma passar do limite, encolhe TODOS os
    gradientes pelo mesmo fator. A direção é mantida; só o tamanho muda."""
    norma = norma_global(parametros)
    if norma > norma_maxima:
        fator = norma_maxima / norma
        for p in parametros:
            p.grad = p.grad * fator
    return norma


class Otimizador:
    """Base comum: guarda os parâmetros e sabe zerar os gradientes."""

    def __init__(self, parametros, taxa_aprendizado, clip_norma=None):
        self.parametros = parametros
        self.taxa = taxa_aprendizado
        self.clip_norma = clip_norma  # None = sem clipping

    def zerar_grad(self):
        # O motor ACUMULA gradientes (+=). Sem zerar, o gradiente de um
        # passo se somaria ao do passo seguinte.
        for p in self.parametros:
            p.grad = np.zeros_like(p.data)

    def _recortar_se_preciso(self):
        if self.clip_norma is not None:
            recortar_gradientes(self.parametros, self.clip_norma)


class SGD(Otimizador):
    """Descida do gradiente estocástica: θ ← θ - η · g"""

    def passo(self):
        self._recortar_se_preciso()
        for p in self.parametros:
            p.data -= self.taxa * p.grad


class Adam(Otimizador):
    """Adam: SGD com momento (média de g) e passo adaptativo (média de g²)."""

    def __init__(self, parametros, taxa_aprendizado=1e-3, clip_norma=None,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parametros, taxa_aprendizado, clip_norma)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # nº de passos dados (usado na correção de viés)
        self.m = [np.zeros_like(p.data) for p in parametros]  # 1º momento
        self.v = [np.zeros_like(p.data) for p in parametros]  # 2º momento

    def passo(self):
        self._recortar_se_preciso()
        self.t += 1
        for i, p in enumerate(self.parametros):
            g = p.grad
            # m ← β1·m + (1-β1)·g        (média móvel do gradiente)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # v ← β2·v + (1-β2)·g²       (média móvel do gradiente ao quadrado)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            # Correção de viés: m e v começam em 0, então nos primeiros
            # passos ficam "puxados" para 0. Dividir por (1-β^t) compensa.
            m_corrigido = self.m[i] / (1 - self.beta1 ** self.t)
            v_corrigido = self.v[i] / (1 - self.beta2 ** self.t)
            # θ ← θ - η · m̂ / (√v̂ + ε)
            p.data -= self.taxa * m_corrigido / (np.sqrt(v_corrigido) + self.eps)


OTIMIZADORES = {
    "sgd": SGD,
    "adam": Adam,
}
