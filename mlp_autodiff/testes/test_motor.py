"""
test_motor.py — Gradient check de CADA operação do motor.

Truque usado em todos os testes: em vez de verificar `op(x).sum()`,
verificamos `(op(x) * C).sum()` com C aleatório fixo. Com `.sum()` puro
o gradiente que chega na operação seria só "uns", o que esconde erros
de transposição/posição. Com C aleatório, cada posição recebe um peso diferente.
"""
import numpy as np
import pytest

from motor import Tensor, _desfazer_broadcast
from checagem import verificar_gradientes

TOLERANCIA = 1e-6
rng = np.random.default_rng(0)


def aleatorio(shape, positivo=False):
    """Tensor aleatório. `positivo=True` para log, sqrt, potência fracionária."""
    if positivo:
        return Tensor(rng.uniform(0.5, 2.0, size=shape))
    return Tensor(rng.normal(size=shape))


def pesos_fixos(shape):
    """Constante C que pondera a saída (ver docstring do módulo)."""
    return Tensor(rng.normal(size=shape))


# Pares de shapes: iguais, broadcasting de linha, de coluna e com escalar.
PARES_DE_SHAPES = [((3, 4), (3, 4)), ((3, 4), (1, 4)), ((3, 4), (3, 1)),
                   ((3, 4), ()), ((), (3, 4)), ((), ())]

OPERACOES_BINARIAS = {
    "soma": lambda a, b: a + b,
    "subtracao": lambda a, b: a - b,
    "produto": lambda a, b: a * b,
    "divisao": lambda a, b: a / b,
}


@pytest.mark.parametrize("nome_op", list(OPERACOES_BINARIAS))
@pytest.mark.parametrize("shapes", PARES_DE_SHAPES)
def test_operacoes_binarias_com_broadcasting(nome_op, shapes):
    op = OPERACOES_BINARIAS[nome_op]
    a = aleatorio(shapes[0])
    b = aleatorio(shapes[1], positivo=True)  # positivo: evita dividir por ~0
    C = pesos_fixos(np.broadcast_shapes(shapes[0], shapes[1]))
    erro = verificar_gradientes(lambda: (op(a, b) * C).sum(), [a, b])
    assert erro < TOLERANCIA


@pytest.mark.parametrize("nome_op", ["radd", "rsub", "rmul", "rtruediv"])
def test_operacoes_reversas_com_escalar_python(nome_op):
    a = aleatorio((3, 4), positivo=True)
    C = pesos_fixos((3, 4))
    funcoes = {
        "radd": lambda: ((2.5 + a) * C).sum(),
        "rsub": lambda: ((2.5 - a) * C).sum(),
        "rmul": lambda: ((2.5 * a) * C).sum(),
        "rtruediv": lambda: ((2.5 / a) * C).sum(),
    }
    assert verificar_gradientes(funcoes[nome_op], [a]) < TOLERANCIA


def test_negacao():
    a = aleatorio((3, 4))
    C = pesos_fixos((3, 4))
    assert verificar_gradientes(lambda: ((-a) * C).sum(), [a]) < TOLERANCIA


@pytest.mark.parametrize("shapes", [((3, 4), (4, 2)), ((1, 5), (5, 1)), ((5, 1), (1, 3))])
def test_matmul(shapes):
    A = aleatorio(shapes[0])
    B = aleatorio(shapes[1])
    C = pesos_fixos((shapes[0][0], shapes[1][1]))
    assert verificar_gradientes(lambda: ((A @ B) * C).sum(), [A, B]) < TOLERANCIA


@pytest.mark.parametrize("p", [2, 3, -1, 0.5, 1.7])
def test_potencia(p):
    x = aleatorio((3, 4), positivo=True)  # positivo: x^0.5 exige x > 0
    C = pesos_fixos((3, 4))
    assert verificar_gradientes(lambda: ((x ** p) * C).sum(), [x]) < TOLERANCIA


OPERACOES_UNARIAS = {
    "exp": lambda x: x.exp(),
    "log": lambda x: x.log(),
    "log_base_10": lambda x: x.log_base(10),
    "log_base_2": lambda x: x.log_base(2),
    "sqrt": lambda x: x.sqrt(),
    "sigmoid": lambda x: x.sigmoid(),
    "tanh": lambda x: x.tanh(),
}


@pytest.mark.parametrize("nome_op", list(OPERACOES_UNARIAS))
@pytest.mark.parametrize("shape", [(), (3, 4)])
def test_operacoes_unarias(nome_op, shape):
    op = OPERACOES_UNARIAS[nome_op]
    x = aleatorio(shape, positivo=True)  # log e sqrt exigem x > 0
    C = pesos_fixos(shape)
    assert verificar_gradientes(lambda: (op(x) * C).sum(), [x]) < TOLERANCIA


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("nome_op", ["sum", "mean"])
def test_soma_e_media(axis, keepdims, nome_op):
    x = aleatorio((3, 4))
    shape_saida = np.sum(x.data, axis=axis, keepdims=keepdims).shape
    C = pesos_fixos(shape_saida)

    def funcao():
        if nome_op == "sum":
            reduzido = x.sum(axis=axis, keepdims=keepdims)
        else:
            reduzido = x.mean(axis=axis, keepdims=keepdims)
        return (reduzido * C).sum()
    assert verificar_gradientes(funcao, [x]) < TOLERANCIA


def test_no_reutilizado_acumula_gradiente():
    # x aparece duas vezes: d(x*x)/dx = 2x. Se o motor SOBRESCREVESSE em vez
    # de ACUMULAR (+=), daria x (metade do valor correto).
    x = Tensor([3.0])
    y = x * x
    y.backward()
    assert np.allclose(x.grad, [6.0])


def test_grafo_em_diamante_ordem_topologica():
    # a é usado por dois caminhos (b e c) que se juntam em d.
    # Só funciona se a ordem topológica estiver certa.
    x = aleatorio((2, 3))
    C = pesos_fixos((2, 3))

    def funcao():
        a = x * 2.0
        b = a.exp()
        c = a * a
        return ((b + c) * C).sum()
    assert verificar_gradientes(funcao, [x]) < TOLERANCIA


def test_desfazer_broadcast_shapes():
    grad = np.ones((5, 3))
    assert _desfazer_broadcast(grad, (1, 3)).tolist() == [[5.0, 5.0, 5.0]]
    assert _desfazer_broadcast(grad, (5, 1)).shape == (5, 1)
    assert _desfazer_broadcast(grad, (3,)).tolist() == [5.0, 5.0, 5.0]
    assert _desfazer_broadcast(grad, ()) == 15.0


def test_sigmoid_primitiva_igual_a_composta():
    # A primitiva é só uma versão numericamente estável de 1/(1+e^-x):
    # em entradas moderadas, valores E gradientes têm que coincidir.
    x1 = aleatorio((3, 4))
    x2 = Tensor(x1.data)
    C = pesos_fixos((3, 4))
    (x1.sigmoid() * C).sum().backward()
    (1.0 / (1.0 + (-x2).exp()) * C).sum().backward()
    assert np.allclose(x1.grad, x2.grad)


def test_sigmoid_tanh_estaveis_em_entradas_grandes():
    # Com a versão composta, e^(1000) = inf e o backward vira NaN.
    x = Tensor([-1000.0, 1000.0])
    (x.sigmoid().sum() + x.tanh().sum()).backward()
    assert np.all(np.isfinite(x.grad))
