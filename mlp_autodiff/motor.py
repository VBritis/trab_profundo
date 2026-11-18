"""
motor.py — Motor de diferenciação automática (modo reverso) em NumPy puro.

Ideia central:
  * Cada operação cria um Tensor de SAÍDA que lembra quem são seus PAIS
    (os tensores de entrada) e guarda uma função `_backward` que sabe
    passar o gradiente da saída para os pais (regra da cadeia local).
  * `backward()` ordena o grafo topologicamente e chama `_backward`
    de trás para frente.

Somente as operações PRIMITIVAS deste arquivo têm derivada escrita.
Tudo o que vem depois (MLP, losses, ativações) é composição delas.
"""
import numpy as np


def _como_tensor(x):
    """Converte números Python / arrays NumPy em Tensor constante (sem pais)."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def _desfazer_broadcast(grad, shape_original):
    """Soma o gradiente nos eixos que o broadcasting 'esticou'.

    Exemplo: b tem shape (1, n) e z = X + b tem shape (batch, n).
    Cada b[0, j] foi usado `batch` vezes, então sua derivada é a SOMA
    das contribuições de todas as linhas: dL/db = soma_linhas(dL/dz).
    """
    # 1) Broadcasting pode criar eixos novos à esquerda: () -> (3, 4).
    #    Esses eixos não existiam no original: somamos e removemos.
    while grad.ndim > len(shape_original):
        grad = grad.sum(axis=0)
    # 2) Eixos que tinham tamanho 1 e foram repetidos: somamos mantendo o eixo.
    for eixo, tamanho in enumerate(shape_original):
        if tamanho == 1 and grad.shape[eixo] != 1:
            grad = grad.sum(axis=eixo, keepdims=True)
    return grad


class Tensor:
    """Um array NumPy + o gradiente dele + a memória de como foi criado."""

    def __init__(self, data, parents=(), nome="", op=""):
        # np.array copia os dados: o otimizador altera .data no lugar e
        # não queremos alterar, sem querer, o array de quem criou o Tensor.
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = parents         # tensores que geraram este
        self._backward = lambda: None  # como passar o gradiente para os pais
        self.nome = nome               # rótulo opcional (ex.: "W1")
        self.op = op                   # operação que criou o tensor (ex.: "+")

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, nome='{self.nome}', op='{self.op}')"

    # ------------------------------------------------------------------
    # Operações elemento a elemento (todas usam _desfazer_broadcast)
    # ------------------------------------------------------------------
    def __add__(self, outro):
        outro = _como_tensor(outro)
        saida = Tensor(self.data + outro.data, (self, outro), op="+")

        # d(a+b)/da = 1 ; d(a+b)/db = 1  -> o gradiente passa direto
        def _backward():
            self.grad += _desfazer_broadcast(saida.grad, self.data.shape)
            outro.grad += _desfazer_broadcast(saida.grad, outro.data.shape)
        saida._backward = _backward
        return saida

    def __sub__(self, outro):
        outro = _como_tensor(outro)
        saida = Tensor(self.data - outro.data, (self, outro), op="-")

        # d(a-b)/da = 1 ; d(a-b)/db = -1
        def _backward():
            self.grad += _desfazer_broadcast(saida.grad, self.data.shape)
            outro.grad += _desfazer_broadcast(-saida.grad, outro.data.shape)
        saida._backward = _backward
        return saida

    def __mul__(self, outro):
        outro = _como_tensor(outro)
        saida = Tensor(self.data * outro.data, (self, outro), op="*")

        # d(a*b)/da = b ; d(a*b)/db = a
        def _backward():
            self.grad += _desfazer_broadcast(saida.grad * outro.data, self.data.shape)
            outro.grad += _desfazer_broadcast(saida.grad * self.data, outro.data.shape)
        saida._backward = _backward
        return saida

    def __truediv__(self, outro):
        outro = _como_tensor(outro)
        saida = Tensor(self.data / outro.data, (self, outro), op="/")

        # d(a/b)/da = 1/b ; d(a/b)/db = -a/b²
        def _backward():
            grad_a = saida.grad / outro.data
            grad_b = -saida.grad * self.data / (outro.data ** 2)
            self.grad += _desfazer_broadcast(grad_a, self.data.shape)
            outro.grad += _desfazer_broadcast(grad_b, outro.data.shape)
        saida._backward = _backward
        return saida

    def __neg__(self):
        saida = Tensor(-self.data, (self,), op="neg")

        # d(-a)/da = -1
        def _backward():
            self.grad += -saida.grad
        saida._backward = _backward
        return saida

    # Versões "reversas": chamadas em `2 + t`, `2 - t`, `2 * t`, `2 / t`.
    # O Python tenta int.__add__(t), falha, e chama t.__radd__(2).
    def __radd__(self, outro):
        return _como_tensor(outro) + self

    def __rsub__(self, outro):
        return _como_tensor(outro) - self

    def __rmul__(self, outro):
        return _como_tensor(outro) * self

    def __rtruediv__(self, outro):
        return _como_tensor(outro) / self

    # ------------------------------------------------------------------
    # Multiplicação de matrizes
    # ------------------------------------------------------------------
    def __matmul__(self, outro):
        outro = _como_tensor(outro)
        saida = Tensor(self.data @ outro.data, (self, outro), op="@")

        # C = A @ B, com A:(n,k), B:(k,m), C:(n,m), G = dL/dC com shape (n,m)
        # dL/dA = G @ B^T   -> (n,m) @ (m,k) = (n,k)  = shape de A
        # dL/dB = A^T @ G   -> (k,n) @ (n,m) = (k,m)  = shape de B
        def _backward():
            self.grad += saida.grad @ outro.data.T
            outro.grad += self.data.T @ saida.grad
        saida._backward = _backward
        return saida

    # ------------------------------------------------------------------
    # Potência, exponencial, logaritmo, raiz
    # ------------------------------------------------------------------
    def __pow__(self, p):
        # p é um número (constante), não um Tensor.
        saida = Tensor(self.data ** p, (self,), op=f"**{p}")

        # d(x^p)/dx = p * x^(p-1)
        def _backward():
            self.grad += saida.grad * p * self.data ** (p - 1)
        saida._backward = _backward
        return saida

    def exp(self):
        saida = Tensor(np.exp(self.data), (self,), op="exp")

        # d(e^x)/dx = e^x  (que é a própria saída)
        def _backward():
            self.grad += saida.grad * saida.data
        saida._backward = _backward
        return saida

    def log(self):
        saida = Tensor(np.log(self.data), (self,), op="log")

        # d(ln x)/dx = 1/x
        def _backward():
            self.grad += saida.grad / self.data
        saida._backward = _backward
        return saida

    def log_base(self, base):
        # Mudança de base: log_b(x) = ln(x) / ln(b). Composta: sem backward novo.
        return self.log() / np.log(base)

    def sqrt(self):
        saida = Tensor(np.sqrt(self.data), (self,), op="sqrt")

        # d(√x)/dx = 1 / (2√x)
        def _backward():
            self.grad += saida.grad / (2.0 * saida.data)
        saida._backward = _backward
        return saida

    # ------------------------------------------------------------------
    # Somatório e média
    # ------------------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        saida = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), op="sum")

        # d(soma)/dx_i = 1 para todo i: cada elemento recebe o gradiente
        # da soma da qual participou. Shape: se o eixo somado foi removido
        # (keepdims=False), recolocamos ele (tamanho 1) e então "esticamos"
        # o gradiente de volta para o shape original de x.
        def _backward():
            g = saida.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self.grad += np.broadcast_to(g, self.data.shape)
        saida._backward = _backward
        return saida

    def mean(self, axis=None, keepdims=False):
        # Média = soma / quantidade. Composta: sem backward novo.
        if axis is None:
            quantidade = self.data.size
        else:
            quantidade = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / quantidade

    # ------------------------------------------------------------------
    # Sigmoid e Tanh como primitivas (estabilidade numérica)
    # ------------------------------------------------------------------
    def sigmoid(self):
        # 1/(1+e^-x) estoura (e^-x = inf) para x muito negativo. A identidade
        # σ(x) = 0.5 * (1 + tanh(x/2)) dá o mesmo valor sem overflow.
        s = 0.5 * (1.0 + np.tanh(0.5 * self.data))
        saida = Tensor(s, (self,), op="sigmoid")

        # dσ/dx = σ(x) * (1 - σ(x))
        def _backward():
            self.grad += saida.grad * saida.data * (1.0 - saida.data)
        saida._backward = _backward
        return saida

    def tanh(self):
        saida = Tensor(np.tanh(self.data), (self,), op="tanh")

        # d(tanh x)/dx = 1 - tanh²(x)
        def _backward():
            self.grad += saida.grad * (1.0 - saida.data ** 2)
        saida._backward = _backward
        return saida

    # ------------------------------------------------------------------
    # Backward: ordenação topológica + regra da cadeia de trás para frente
    # ------------------------------------------------------------------
    def backward(self):
        # Ordenação topológica por DFS: um nó só entra na lista DEPOIS de
        # todos os seus pais. Percorrendo a lista ao contrário, cada nó só
        # repassa o gradiente quando já recebeu TODAS as contribuições.
        ordem = []
        visitados = set()

        def visitar(no):
            if no in visitados:
                return
            visitados.add(no)
            for pai in no.parents:
                visitar(pai)
            ordem.append(no)

        visitar(self)
        self.grad = np.ones_like(self.data)  # dL/dL = 1
        for no in reversed(ordem):
            no._backward()


def imprimir_grafo(tensor, nivel=0):
    """Imprime em texto a árvore do grafo computacional (para explicar no quadro)."""
    rotulo = tensor.op if tensor.op else "folha"
    nome = f" '{tensor.nome}'" if tensor.nome else ""
    print("    " * nivel + f"[{rotulo}]{nome} shape={tensor.data.shape}")
    for pai in tensor.parents:
        imprimir_grafo(pai, nivel + 1)
