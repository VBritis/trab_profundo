# MLP com Diferenciação Automática (NumPy puro)

Projeto 1 — implementação de um **motor de diferenciação automática** (modo reverso) e de um **MLP** construído
sobre ele, sem nenhum framework de deep learning. O backward da rede vem exclusivamente do motor: as únicas
derivadas escritas à mão são as das operações primitivas em `motor.py`.

## Como rodar

```powershell
# na pasta Trab_Profundo (o ambiente virtual .venv já existe)
.\.venv\Scripts\Activate.ps1
cd mlp_autodiff
python -m pytest -q                 # 89 testes, ~3 s
jupyter notebook                    # abrir os notebooks 01, 02, 03
```
Instalação do zero: `pip install -r requirements.txt`. Na primeira execução os datasets são baixados para
`dados_cache/` (o Fashion-MNIST demora ~30 s uma única vez; depois fica em `.npz`).

Todos os notebooks têm `MODO_RAPIDO = True` no topo (subconjunto dos dados e menos épocas): cada um roda em
**menos de 40 s**. Com `False`, usam todos os dados e mais épocas.

## Estrutura

```
motor.py          classe Tensor + operações primitivas e seus backward
checagem.py       verificação de gradientes por diferenças finitas
rede.py           inicializações, ativações, losses, Camada, MLP
exportacao.py     salvar/carregar o modelo em JSON (topologia + pesos + padronização)
otimizadores.py   SGD, Adam, gradient clipping
treino.py         loop de treino, mini-lotes, métricas, histórico, histogramas, early stopping
dados.py          California Housing e Fashion-MNIST: carregar, dividir 70/15/15, padronizar
graficos.py       curvas de custo/métrica, comparações, matriz de confusão, tabela
graficos_gradientes.py  histogramas, norma do gradiente por camada, profundidade, ReLU mortos
testes/           test_motor.py (cada operação) e test_rede.py (provas ponta a ponta)
01_regressao.ipynb       California Housing
02_classificacao.ipynb   Fashion-MNIST (sigmoid+BCE × softmax+CE, SGD × Adam, He × Xavier)
03_gradientes.ipynb      dissipação e explosão do gradiente: prever, medir, corrigir
modelos/                 modelos exportados (.json legível)
ROTEIRO_DEFESA.md        explicação de cada arquivo + perguntas prováveis
MUDANCAS_PROVAVEIS.md    mudanças prováveis com o código pronto (todas testadas)
```

## Checklist do enunciado

| Requisito | Onde |
|---|---|
| Soma, subtração, produto, divisão (escalar/matriz, broadcasting) | `motor.py` (`__add__`... + reversos) |
| Multiplicação de matrizes | `Tensor.__matmul__` |
| Exponenciação, radiciação, logaritmos | `exp`, `__pow__`, `sqrt`, `log`, `log_base` |
| Somatório | `Tensor.sum(axis, keepdims)` (e `mean` composto) |
| Corretude por diferenças finitas | `testes/test_motor.py` (cada operação, vários shapes) |
| Nº de camadas e unidades | `config["camadas_ocultas"]` |
| Sigmoid, Tanh, ReLU, Leaky ReLU, Linear | `rede.ATIVACOES` (+ softmax) |
| BCE e MSE | `rede.PERDAS` (+ entropia cruzada) |
| SGD e Adam | `otimizadores.py` |
| He, Xavier, Normal, Uniforme | `rede.INICIALIZACOES` |
| Exportar / importar e inferir | `exportacao.py` (notebooks 01 e 02 mostram o round-trip) |
| Classificação e regressão | Fashion-MNIST (02) e California Housing (01) |
| Gradiente que se dissipa / explode | notebook 03 |
| Histogramas de ativações e gradientes | `graficos_gradientes.plot_histogramas` (01, 02, 03) |
| Custo e métrica vs. época (treino e validação) | `graficos.plot_curvas` (01, 02) |

## Resultados (modo rápido, conjunto de teste)

| Problema | Modelo | Resultado |
|---|---|---|
| California Housing | MLP [64, 32] ReLU, Adam | R² ≈ 0,74 · MAE ≈ 0,40 (US$ 40 mil) |
| Fashion-MNIST | MLP [128, 64] ReLU, softmax + CE, Adam | acurácia ≈ 85% |
| Fashion-MNIST | idem, sigmoid + BCE | acurácia ≈ 84% |

Experimento de gradientes (10 camadas ocultas): com Sigmoid + Normal(0,1) a norma do gradiente da 1ª camada é
~10⁻⁸ da última (previsão pela sigmoid: 0,25¹⁰ ≈ 10⁻⁶) e o R² fica em 0; com ReLU + Normal(1,0) o sinal
cresce 4,3× por camada (previsto: 4×) e o custo vira NaN na 3ª iteração. ReLU + He (e He + clipping) corrigem.

## Por que California Housing e não Boston Housing?

O Boston Housing foi **removido do scikit-learn na versão 1.2**. Uma de suas variáveis (`B`) foi construída
a partir da proporção de moradores negros de cada bairro, com a suposição de que a segregação racial teria
efeito no preço das casas — um viés racial embutido no próprio dado. O California Housing é o substituto
recomendado pela documentação do scikit-learn.

## Regras seguidas

- Sem PyTorch/TensorFlow/JAX/autograd em nenhuma etapa (nem para conferir gradientes).
- scikit-learn só para baixar os dados e fazer o split.
- `float64` em tudo; seeds fixas (`np.random.default_rng(seed)`).
