# Mudanças prováveis (para fazer ao vivo)

Cada mudança diz **arquivo**, **onde mexer** e **o código**. Todas foram aplicadas numa cópia do projeto e
testadas (gradient check, `pytest` ou treino curto) antes de entrar aqui.

Depois de qualquer mudança: `python -m pytest -q` (≈ 3 s).

> Regra de ouro para explicar: **se a mudança é composta só por operações do motor, ela NÃO precisa de
> backward.** Só uma operação nova *no motor* (mudança 2) precisa de derivada escrita.

---

## 1. Nova ativação (Softplus, SiLU/Swish, ELU)

**Arquivo:** `rede.py` — logo depois de `def linear(x)`, e registrar no dicionário `ATIVACOES`.

```python
def softplus(x):
    # softplus(x) = ln(1 + e^x)  -> versão "suave" da ReLU. Só composição.
    return (1.0 + x.exp()).log()


def silu(x):
    # SiLU / Swish: x · σ(x). Só composição.
    return x * x.sigmoid()


def elu(x, alfa=1.0):
    # x se x > 0 ; alfa·(e^x - 1) se x <= 0. Máscara constante, como na ReLU.
    mascara_positivo = Tensor((x.data > 0) * 1.0)
    parte_negativa = x * (1.0 - mascara_positivo)  # zera os positivos (evita e^x gigante)
    return x * mascara_positivo + alfa * (parte_negativa.exp() - 1.0)
```

No dicionário `ATIVACOES`:
```python
    "softplus": softplus,
    "silu": silu,
    "elu": elu,
```
**Usar:** `config["ativacao_oculta"] = "silu"`. **Testar:** acrescente o nome na lista do
`@pytest.mark.parametrize("ativacao", [...])` em `testes/test_rede.py`.

---

## 2. Nova operação no motor (`sin`, `abs`) + gradient check

**Arquivo:** `motor.py` — dentro da classe `Tensor`, perto de `log`/`sqrt` (mesmo padrão de toda primitiva).

```python
    def sin(self):
        saida = Tensor(np.sin(self.data), (self,), op="sin")

        # d(sen x)/dx = cos x
        def _backward():
            self.grad += saida.grad * np.cos(self.data)
        saida._backward = _backward
        return saida

    def abs(self):
        saida = Tensor(np.abs(self.data), (self,), op="abs")

        # d|x|/dx = sinal(x)   (não é derivável em x = 0)
        def _backward():
            self.grad += saida.grad * np.sign(self.data)
        saida._backward = _backward
        return saida
```

**Teste** — `testes/test_motor.py`, no dicionário `OPERACOES_UNARIAS`:
```python
    "sin": lambda x: x.sin(),
    "abs": lambda x: x.abs(),
```
e (opcional) um teste do `abs` com negativos, longe de 0:
```python
def test_abs_com_entradas_negativas():
    # Sorteia longe de 0 (|x| >= 0.5), onde |x| é derivável.
    x = Tensor(rng.uniform(0.5, 2.0, size=(3, 4)) * rng.choice([-1, 1], size=(3, 4)))
    C = pesos_fixos((3, 4))
    assert verificar_gradientes(lambda: (x.abs() * C).sum(), [x]) < TOLERANCIA
```

O padrão para **qualquer** operação nova: (1) calcula `saida` com NumPy; (2) escreve a derivada no comentário;
(3) `_backward` faz `self.grad += saida.grad * derivada` (sempre `+=`); (4) se for binária com broadcasting, use
`_desfazer_broadcast`.

---

## 3. Nova loss (MAE, Huber)

**Arquivo:** `rede.py` — antes de `PERDAS = {`, e registrar no dicionário.

```python
def mae(predicao, alvo):
    # média de |pred - alvo|. |d| = d · sinal(d), com sinal constante (truque da ReLU).
    diferenca = predicao - alvo
    sinal = Tensor(np.sign(diferenca.data))
    return (diferenca * sinal).mean()


def huber(predicao, alvo, delta=1.0):
    # |d| <= delta: 0.5·d²  (quadrática) ; |d| > delta: delta·(|d| - 0.5·delta)  (linear)
    diferenca = predicao - alvo
    pequeno = Tensor((np.abs(diferenca.data) <= delta) * 1.0)  # máscara constante
    sinal = Tensor(np.sign(diferenca.data))
    quadratica = 0.5 * diferenca ** 2
    linear_ = delta * (diferenca * sinal - 0.5 * delta)
    return (pequeno * quadratica + (1.0 - pequeno) * linear_).mean()
```
```python
    "mae": mae,
    "huber": huber,
```
**Usar:** `config_treino["perda"] = "huber"`. (Se a mudança 2 já foi feita, o MAE pode ser só
`(predicao - alvo).abs().mean()`.)

---

## 4. SGD com momento / RMSProp

**Arquivo:** `otimizadores.py` — antes de `OTIMIZADORES = {`, e registrar no dicionário.

```python
class SGDMomento(Otimizador):
    """v ← μ·v - η·g ;  θ ← θ + v   (a "velocidade" acumula a direção)"""

    def __init__(self, parametros, taxa_aprendizado=0.01, clip_norma=None, momento=0.9):
        super().__init__(parametros, taxa_aprendizado, clip_norma)
        self.momento = momento
        self.velocidade = [np.zeros_like(p.data) for p in parametros]

    def passo(self):
        self._recortar_se_preciso()
        for i, p in enumerate(self.parametros):
            self.velocidade[i] = self.momento * self.velocidade[i] - self.taxa * p.grad
            p.data += self.velocidade[i]


class RMSProp(Otimizador):
    """s ← β·s + (1-β)·g² ;  θ ← θ - η·g / (√s + ε)   (Adam sem o momento)"""

    def __init__(self, parametros, taxa_aprendizado=1e-3, clip_norma=None, beta=0.9, eps=1e-8):
        super().__init__(parametros, taxa_aprendizado, clip_norma)
        self.beta = beta
        self.eps = eps
        self.media_g2 = [np.zeros_like(p.data) for p in parametros]

    def passo(self):
        self._recortar_se_preciso()
        for i, p in enumerate(self.parametros):
            self.media_g2[i] = self.beta * self.media_g2[i] + (1 - self.beta) * p.grad ** 2
            p.data -= self.taxa * p.grad / (np.sqrt(self.media_g2[i]) + self.eps)
```
```python
    "sgd_momento": SGDMomento,
    "rmsprop": RMSProp,
```
**Usar:** `config_treino["otimizador"] = "sgd_momento"`.

---

## 5. Weight decay / L2

**Jeito 1 — já existe (L2 na loss):** `config_treino["lambda_l2"] = 1e-3`. Está em
`treino.penalidade_l2`: `perda + λ·Σ W²`. Nenhuma derivada escrita.

**Jeito 2 — weight decay dentro do otimizador (SGD):**

`otimizadores.py`, em `Otimizador.__init__`, depois de `self.clip_norma = ...`:
```python
        self.weight_decay = 0.0       # λ do weight decay (0 = desligado)
```
`otimizadores.py`, em `SGD.passo`, troque a linha da atualização por:
```python
            # weight decay: θ ← θ - η·(g + λ·θ)  (puxa os pesos para 0 a cada passo)
            p.data -= self.taxa * (p.grad + self.weight_decay * p.data)
```
`treino.py`, em `CONFIG_TREINO_PADRAO`:
```python
    "weight_decay": 0.0,         # weight decay no SGD (0 = desligado)
```
`treino.py`, em `treinar`, logo depois de criar o `otimizador`:
```python
    otimizador.weight_decay = cfg["weight_decay"]
```
(No SGD, os dois jeitos são equivalentes: a derivada de λ/2·W² é λ·W. No Adam, não são — daí o AdamW.)

---

## 6. Dropout

**Arquivo:** `rede.py` — 3 lugares; e `treino.py` — 2 linhas.

`rede.py`, antes de `ATIVACOES = {`:
```python
def dropout(x, taxa, rng):
    # "Inverted dropout": zera cada unidade com probabilidade `taxa` e divide as
    # que sobram por (1 - taxa), para a média do sinal não mudar.
    # A máscara é CONSTANTE: o motor deriva sozinho (como na ReLU).
    mascara = Tensor((rng.random(x.data.shape) >= taxa) / (1.0 - taxa))
    return x * mascara
```
`rede.py`, no fim de `MLP.__init__`:
```python
        self.taxa_dropout = config.get("dropout", 0.0)
        self.rng_dropout = np.random.default_rng(config.get("seed", 0))
        self.treinando = False     # dropout só fica ligado durante o treino
```
`rede.py`, em `MLP.forward`, troque o laço por:
```python
        for i, camada in enumerate(self.camadas):
            x = camada.forward(x)
            eh_oculta = i < len(self.camadas) - 1
            if self.treinando and eh_oculta and self.taxa_dropout > 0:
                x = dropout(x, self.taxa_dropout, self.rng_dropout)
            self.ativacoes.append(x)
```
`treino.py`, em `treinar`: `rede.treinando = True` logo antes do `for X_lote, Y_lote in ...` e
`rede.treinando = False` logo depois do laço (antes de avaliar).

**Usar:** `config["dropout"] = 0.3`.

---

## 7. Learning rate decay

**Arquivo:** `treino.py`.

Em `CONFIG_TREINO_PADRAO`:
```python
    "decaimento_taxa": 1.0,      # a cada época: taxa ← taxa · decaimento (1.0 = sem decaimento)
```
Em `treinar`, logo depois de `historico["norma_grad"].append(...)`:
```python
        otimizador.taxa = otimizador.taxa * cfg["decaimento_taxa"]  # learning rate decay
```
**Usar:** `config_treino["decaimento_taxa"] = 0.95` (decaimento exponencial: η_época = η₀ · 0,95^época).

---

## 8. Imprimir o gradiente de um peso específico

**Em uma célula do notebook** (não precisa mexer em arquivo):
```python
from treino import calcular_perda
X_lote, Y_lote = dados["X_treino"][:16], dados["Y_treino"][:16]

for p in rede.parametros():                   # 1) zera
    p.grad = np.zeros_like(p.data)
perda, _ = calcular_perda(rede, X_lote, Y_lote, "mse")
perda.backward()                              # 2) backward
print("dL/dW1[0, 2] pelo motor:      ", rede.camadas[0].W.grad[0, 2])

h = 1e-6                                      # 3) confere por diferença finita
W = rede.camadas[0].W
W.data[0, 2] += h
perda_mais = calcular_perda(rede, X_lote, Y_lote, "mse")[0].data
W.data[0, 2] -= 2 * h
perda_menos = calcular_perda(rede, X_lote, Y_lote, "mse")[0].data
W.data[0, 2] += h
print("dL/dW1[0, 2] por dif. finitas:", (perda_mais - perda_menos) / (2 * h))
```
Variações: gradiente do bias → `rede.camadas[0].b.grad`; gradiente de uma **ativação** →
`rede.ativacoes[0].grad` (vem de graça do motor). Para imprimir **durante o treino**: em `treino.py`, depois de
`perda.backward()`, coloque `print(rede.camadas[0].W.grad[0, 2])`.

---

## 9. Mudar a arquitetura só pela configuração

Nenhum arquivo muda — só o dicionário:
```python
config = {
    "n_entrada": 8,
    "camadas_ocultas": [32, 16, 8],   # 3 camadas ocultas (lista vazia = regressão linear)
    "n_saida": 1,
    "ativacao_oculta": "leaky_relu",
    "ativacao_saida": "linear",
    "inicializacao": "he",
    "seed": 7,
}
rede = MLP(config)
```
Trocar otimizador / loss / clipping também é só configuração: `config_treino["otimizador"] = "sgd"`,
`["perda"] = "bce"`, `["clip_norma"] = 1.0`.

---

## 10. Early stopping com outra paciência / outra métrica

**Outra paciência:** só configuração — `config_treino["paciencia"] = 3`.

**Outra métrica** (parar pela métrica de validação em vez do custo) — `treino.py`:

Em `CONFIG_TREINO_PADRAO`:
```python
    "criterio_parada": "perda_val",  # ou "metrica_val" (maior é melhor)
```
Em `treinar`, no bloco do early stopping, troque `if perda_val < melhor_perda_val:` e a linha de baixo por:
```python
            # Sempre "menor é melhor": para a métrica (maior é melhor) usamos o negativo.
            if cfg["criterio_parada"] == "metrica_val":
                valor_atual = -metrica_val
            else:
                valor_atual = perda_val
            if valor_atual < melhor_perda_val:
                melhor_perda_val = valor_atual
```

---

## 11. (Extra) Nova inicialização (LeCun)

**Arquivo:** `rede.py` — antes de `INICIALIZACOES = {`, e registrar `"lecun": init_lecun,`.
```python
def init_lecun(fan_in, fan_out, rng, escala):
    # N(0, √(1/fan_in)). Usada com SELU / tanh.
    return rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
```

---

## Outras perguntas rápidas "mostre…"

| Pedido | Onde |
|---|---|
| "Mostre o grafo dessa expressão" | `from motor import imprimir_grafo; imprimir_grafo(saida)` |
| "Mostre que o gradiente está certo" | `checagem.verificar_gradientes(funcao, [tensores])` |
| "Mude o tamanho do lote / épocas / taxa" | chaves `tamanho_lote`, `epocas`, `taxa_aprendizado` em `config_treino` |
| "Ligue o gradient clipping" | `config_treino["clip_norma"] = 1.0` |
| "Rode tudo mais rápido / com todos os dados" | `MODO_RAPIDO = True / False` no topo do notebook |
| "Abra o arquivo exportado" | `modelos/*.json` (texto legível) |
