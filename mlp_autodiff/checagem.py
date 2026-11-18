"""
checagem.py — Verificação de gradientes por diferenças finitas centrais.

Para cada elemento x_i de um tensor:
    df/dx_i ≈ [ f(x_i + h) - f(x_i - h) ] / (2h)
Comparamos esse número com o gradiente que o motor calculou no backward.
"""
import numpy as np


def gradiente_numerico(funcao, tensor, h=1e-6):
    """Aproxima d(funcao)/d(tensor) mexendo em um elemento por vez.

    `funcao` não recebe argumentos e devolve um Tensor ESCALAR; ela é
    reavaliada do zero a cada perturbação (o grafo é reconstruído).
    """
    grad = np.zeros_like(tensor.data)
    for indice in np.ndindex(tensor.data.shape):
        original = tensor.data[indice]

        tensor.data[indice] = original + h
        f_mais = funcao().data
        tensor.data[indice] = original - h
        f_menos = funcao().data

        tensor.data[indice] = original  # devolve o valor original
        grad[indice] = (f_mais - f_menos) / (2 * h)
    return grad


def erro_relativo(a, b):
    """max|a - b| / max(|a| + |b|) — independe da escala do gradiente."""
    numerador = np.max(np.abs(a - b))
    denominador = max(np.max(np.abs(a) + np.abs(b)), 1e-12)
    return numerador / denominador


def verificar_gradientes(funcao, tensores, h=1e-6):
    """Roda o backward do motor e compara com diferenças finitas.

    Devolve o MAIOR erro relativo entre todos os tensores.
    """
    for t in tensores:
        t.grad = np.zeros_like(t.data)  # zera antes (gradientes acumulam!)
    saida = funcao()
    saida.backward()

    pior_erro = 0.0
    for t in tensores:
        numerico = gradiente_numerico(funcao, t, h)
        pior_erro = max(pior_erro, erro_relativo(t.grad, numerico))
    return pior_erro
