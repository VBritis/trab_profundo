# Projeto 1 — MLP com Diferenciação Automática (NumPy puro)

> **Como usar:** coloque este arquivo na raiz de uma pasta vazia do projeto e diga ao Claude Code:
> *"Leia PROMPT_CLAUDE_CODE.md e execute a Fase 0. Pare ao final de cada fase e espere meu OK."*

---

## 0. Contexto (leia antes de escrever qualquer código)

Sou aluno e vou **apresentar este trabalho ao professor em aula**. Ele vai:
1. pedir que eu explique **qualquer linha** do código;
2. pedir **mudanças ao vivo** (ex.: "adicione uma ativação nova", "troque o otimizador", "mostre o gradiente desse peso").

Portanto, **a prioridade nº 1 é legibilidade**, acima de desempenho, elegância ou generalidade. Eu preciso conseguir reescrever qualquer parte de cabeça. Se houver um jeito "esperto" e um jeito óbvio, use o óbvio.

O enunciado completo está resumido na seção 2. Os "plus" (seção 4) são o diferencial do trabalho, mas **nunca podem complicar o núcleo** (motor + MLP).

---

## 1. Regras de código (obrigatórias)

**Proibido**
- PyTorch, TensorFlow, JAX, autograd, ou qualquer framework de DL — **nem para conferir gradientes**. O enunciado diz "em nenhuma etapa".
- Derivadas escritas à mão por camada. Só as **operações primitivas do motor** têm `backward`; o MLP inteiro é composto delas.
- Metaclasses, decorators customizados, herança com mais de 1 nível, `**kwargs` genéricos, list comprehensions aninhadas, one-liners crípticos, programação "mágica".
- Copiar o micrograd do Karpathy. Pode ter a mesma ideia (é o padrão da área), mas o código deve ser original, **vetorizado** (micrograd é escalar) e com nomes próprios.

**Permitido**
- `numpy`, `matplotlib`, `json`. `scikit-learn` **apenas** para baixar datasets e fazer o split treino/val/teste — nunca para modelar.

**Estilo**
- Comentários e docstrings **em português**.
- Toda função `backward` de primitiva deve ter, num comentário acima, **a derivada em notação matemática** (ex.: `# d(a*b)/da = b ; d(a*b)/db = a`) e o raciocínio do formato (shape) quando houver matriz.
- Cada arquivo com no máximo ~200 linhas. Funções curtas (idealmente < 25 linhas).
- Nomes descritivos e consistentes: `Tensor`, `data`, `grad`, `parents`, `_backward`.
- Sem otimizações prematuras. `float64` em todo lugar (facilita o gradient check).
- Seeds fixas (`np.random.default_rng(seed)`) em tudo, para reprodutibilidade.

---

## 2. Requisitos do enunciado (checklist — nada pode faltar)

**Motor de autodiff** (escalares e matrizes NumPy, com backward correto):
- [ ] soma, subtração, produto, divisão (escalar∘escalar, escalar∘matriz e matriz∘matriz com broadcasting), incluindo `__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__neg__`
- [ ] multiplicação de matrizes (`@`)
- [ ] exponenciação (`exp` e `x ** p`), radiciação (`sqrt`), logaritmo (`log`; e `log` em outra base composto por mudança de base)
- [ ] somatório (`sum` com `axis` e `keepdims`); `mean` composto a partir de `sum`
- [ ] verificação por **diferenças finitas** para TODAS as operações

**MLP** — hiperparâmetros configuráveis:
- [ ] nº de camadas ocultas e nº de unidades por camada
- [ ] ativações: Sigmoid, Tanh, ReLU, Leaky ReLU, Linear
- [ ] custo: BCE e MSE
- [ ] otimizador: SGD e Adam
- [ ] inicialização: He, Xavier, Normal, Uniforme
- [ ] **exportar** arquivo com topologia + pesos; **importar** e inferir sem treinar

**Dados**: ≥ 1 problema de classificação e ≥ 1 de regressão.

**Experimento**: provocar e evidenciar **gradiente que se dissipa** ou **explosivo** nos gráficos.

**Monitoramento** (gráficos):
- [ ] histograma das ativações e dos gradientes ao longo do treino
- [ ] custo vs. época (treino e validação)
- [ ] métrica vs. época (treino e validação)

---

## 3. Armadilhas que a maioria vai errar (trate explicitamente e comente no código)

1. **Broadcasting no backward.** Em `W @ x + b`, o `b` tem shape `(1, n)` mas o gradiente chega `(batch, n)`. É preciso **somar o gradiente nos eixos que foram "esticados"**. Crie uma função auxiliar única e bem comentada, `_desfazer_broadcast(grad, shape_original)`, usada por todas as operações elemento-a-elemento. Teste com shapes diferentes no gradient check.
2. **Nó usado mais de uma vez** (ex.: `x * x`, ou `p` aparecendo duas vezes na BCE). Os gradientes devem **acumular** (`+=`), nunca sobrescrever.
3. **Ordem do backward.** Use **ordenação topológica** do grafo (DFS simples), não recursão ingênua — senão um nó propaga gradiente antes de receber todas as contribuições.
4. **`zero_grad`.** Gradientes acumulam entre iterações se não forem zerados. Deixe isso explícito no loop de treino.
5. **Gradient check em pontos não-diferenciáveis.** ReLU em `x = 0` quebra a diferença finita. Nos testes, sorteie entradas longe de 0 e explique isso num comentário.
6. **Estabilidade numérica.**
   - `log(0)` na BCE → usar `log(p + eps)` e `log(1 - p + eps)` com `eps = 1e-7` (composto só com primitivas).
   - Sigmoid e Tanh compostas com `exp` geram `inf/inf = NaN` no backward para entradas grandes (justamente no experimento de gradiente explosivo!). Por isso **implemente `sigmoid` e `tanh` como primitivas do motor**, com versão estável. Nos testes, mostre que a primitiva `sigmoid` dá o mesmo gradiente que a versão composta `1 / (1 + exp(-x))` em entradas moderadas — prova de que a primitiva é só uma otimização numérica, não uma "derivada manual por camada".
   - **ReLU e Leaky ReLU NÃO precisam de backward próprio**: `relu(x) = x * mascara`, onde `mascara = Tensor((x.data > 0) * 1.0)` é constante. O autodiff deriva sozinho. Comente o porquê.
7. **BCE com várias classes.** BCE é binária. Para multiclasse, use one-hot + uma sigmoid por saída (cada saída responde "é da classe k?"). Comente isso — o professor vai perguntar.
8. **Pré-processamento no arquivo exportado.** Se a entrada (e o alvo, na regressão) foi padronizada, a média e o desvio **precisam ir junto no arquivo exportado**, senão a inferência após importar fica errada. Quase todo mundo esquece.
9. **Boston Housing** foi removido do scikit-learn (versão 1.2) por conter uma variável com viés racial. Use **California Housing** e explique o motivo no README (mostra senso crítico).

---

## 4. Os "plus" (o diferencial) — em ordem de prioridade

Faça na ordem. Todos são pequenos e **isolados** do núcleo.

### Plus 1 — Provas de corretude ponta a ponta (barato e muito convincente)
- **Equivalência com regressão linear:** um MLP com **0 camadas ocultas**, ativação Linear e MSE é uma regressão linear. Treine com Adam e compare os pesos com a solução fechada `np.linalg.lstsq`. Devem coincidir (tolerância pequena). Isso valida motor + otimizador + loss de uma vez.
- **Overfit de um mini-lote:** com 16 exemplos, a rede deve levar a loss a ~0. Se não leva, há bug.
- **Gradient check da rede inteira:** além de cada operação, verifique por diferenças finitas o gradiente da loss de um MLP pequeno (2 camadas, cada ativação) em relação a todos os pesos.
- **Round-trip de exportação:** exportar → importar → as predições devem ser **idênticas** (`np.allclose`).

### Plus 2 — Gradiente que se dissipa/explode "como cientista"
Não basta mostrar o problema; **prever, medir e corrigir**:
- **Prever:** a derivada máxima da sigmoid é 0,25; numa rede de L camadas o gradiente nas primeiras camadas tende a cair ~0,25^L. Escreva isso no notebook.
- **Medir:** gráfico da **norma do gradiente por camada** (eixo y em log) — tanto ao longo das épocas quanto em função da profundidade. Rede funda (ex.: 10 camadas) com Sigmoid + init Normal de desvio pequeno → linhas das primeiras camadas despencam.
- **Explosão:** rede funda com init Normal de desvio grande (ex.: 1.0) → norma cresce até `inf`/NaN; a loss "estoura" no gráfico.
- **Corrigir:** mesma rede com ReLU + He (dissipação) e com init correta + **gradient clipping por norma** (explosão; ~5 linhas no otimizador). Gráfico **antes × depois** lado a lado.
- **Neurônios ReLU mortos:** % de unidades com ativação zero em todo o lote, por camada, ao longo do treino (sai de graça dos dados do histograma).

### Plus 3 — Vitrine do autodiff (mostra que eu entendi o PORQUÊ do motor)
Recursos implementados **só compondo operações existentes, sem escrever nenhum backward novo**:
- **Regularização L2:** `loss = loss + lambda * soma(W**2)` — 2 linhas.
- **Softmax + Entropia Cruzada** (com o truque de subtrair o máximo para estabilidade), comparada com sigmoid+BCE na classificação.
Deixe um comentário dizendo explicitamente: "nenhuma derivada foi escrita aqui".

### Plus 4 — Kit de defesa (o mais importante para a apresentação)
- `ROTEIRO_DEFESA.md`: para cada arquivo, o que cada bloco faz, em linguagem simples; + as **20 perguntas mais prováveis** do professor com resposta curta (ex.: por que ordenação topológica? por que somar no broadcasting? por que He para ReLU e Xavier para Tanh? o que é a correção de viés do Adam? por que zerar gradientes? por que validação separada do teste?).
- `MUDANCAS_PROVAVEIS.md`: lista de **mudanças que o professor provavelmente vai pedir**, cada uma com *arquivo*, *onde mexer* e *o código da mudança* (de 1 a 10 linhas). No mínimo:
  1. nova ativação (Softplus, ELU, Swish/SiLU)
  2. nova operação no motor (ex.: `sin`, `abs`) + seu gradient check
  3. nova loss (MAE, Huber)
  4. SGD com momentum / RMSProp
  5. weight decay / L2
  6. dropout
  7. learning rate decay
  8. imprimir o gradiente de um peso específico
  9. mudar a arquitetura só pela configuração
  10. early stopping com outra paciência / outra métrica
- **Modo rápido:** todo experimento roda em **menos de 60 segundos** com `MODO_RAPIDO = True` (subconjunto dos dados, menos épocas), para eu poder alterar e rodar de novo na frente do professor.

### Plus 5 — Detalhes de engenharia (só se sobrar tempo)
- Early stopping guardando o melhor modelo na validação (reaproveita a exportação).
- `imprimir_grafo(tensor)`: imprime em texto a árvore do grafo computacional de uma expressão pequena (útil para explicar o motor no quadro).
- Tabela final com resultados no conjunto de **teste** (usado uma única vez, no final).

---

## 5. Estrutura de arquivos

```
mlp_autodiff/
├── motor.py          # classe Tensor + operações primitivas e seus backward
├── rede.py           # inicializações, ativações, losses, Camada, MLP, salvar/carregar
├── otimizadores.py   # SGD, Adam (+ clipping)
├── treino.py         # loop de treino, mini-lotes, métricas, histórico, coleta p/ histogramas
├── graficos.py       # todas as funções de plot
├── dados.py          # carregar, dividir e padronizar os datasets
├── testes/
│   ├── test_motor.py     # gradient check de cada operação (com broadcasting)
│   └── test_rede.py      # gradient check da rede, OLS, overfit, round-trip
├── 01_regressao.ipynb        # California Housing
├── 02_classificacao.ipynb    # Fashion-MNIST (sigmoid+BCE e softmax+CE)
├── 03_gradientes.ipynb       # dissipação/explosão: prever, medir, corrigir
├── modelos/                  # arquivos exportados
├── README.md
├── ROTEIRO_DEFESA.md
└── MUDANCAS_PROVAVEIS.md
```

Notebooks só **chamam** funções dos `.py` e mostram gráficos; nada de lógica escondida em célula.

---

## 6. Decisões de design (siga estas, para manter o código simples)

**Tensor**
```python
class Tensor:
    def __init__(self, data, parents=(), nome=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents = parents        # tensores que geraram este
        self._backward = lambda: None # como passar o gradiente para os pais
        self.nome = nome
```
- Cada operação cria o `Tensor` de saída e define `_backward` como uma função interna curta.
- `backward()`: ordenação topológica → `grad` da saída = 1 → percorre ao contrário chamando `_backward`.
- Escalares Python são convertidos para `Tensor` automaticamente (função `_como_tensor`).

**Camada e MLP**
- `Camada(n_entrada, n_saida, ativacao, inicializacao)` com `W` e `b` (`b` com shape `(1, n_saida)`, zeros).
- Ativação é uma **string** (`"relu"`, `"tanh"`...) mapeada num dicionário para funções — fácil de exportar em JSON e fácil de estender.
- `MLP(config)` onde `config` é um **dicionário simples**:
  ```python
  config = {
      "n_entrada": 8,
      "camadas_ocultas": [64, 32],
      "ativacao_oculta": "relu",
      "ativacao_saida": "linear",
      "inicializacao": "he",
      "seed": 42,
  }
  ```
- A rede guarda a última ativação de cada camada (`self.ativacoes`) para os histogramas — os gradientes dessas ativações já vêm de graça do autodiff (`.grad` dos tensores intermediários). Comente isso: é uma vantagem direta do motor.

**Inicializações** (bias sempre zero; comentar a fórmula e a intuição de cada uma)
- He: `N(0, sqrt(2 / fan_in))` — para ReLU
- Xavier: `N(0, sqrt(2 / (fan_in + fan_out)))` — para Tanh/Sigmoid
- Normal: `N(0, desvio)` com `desvio` configurável (usado nos experimentos de dissipação/explosão)
- Uniforme: `U(-limite, limite)` com `limite` configurável

**Otimizadores**: classes com `passo()` e `zerar_grad()`. Adam com correção de viés, comentando cada linha com a fórmula.

**Exportação**: **um único arquivo JSON** legível (o professor pode abrir no editor) contendo: versão, `config`, nome da loss, pesos como listas, **estatísticas de padronização** (entrada e alvo), e métricas finais. `carregar_modelo(caminho)` reconstrói a rede e já aplica a padronização na inferência.

**Histogramas**: coletar ativações e gradientes (das ativações e dos pesos) em um **lote fixo de validação** a cada N épocas — lote fixo deixa os histogramas comparáveis ao longo do tempo. Plotar em grade (linhas = épocas, colunas = camadas).

**Dados**
- Regressão: California Housing (`sklearn.datasets.fetch_california_housing`). Padronizar X e y. Métricas: MSE, **R²** e MAE na escala original.
- Classificação: Fashion-MNIST (`fetch_openml("Fashion-MNIST")`), pixels / 255, one-hot. Métrica: acurácia (+ matriz de confusão no teste).
- Split fixo treino / validação / teste (ex.: 70/15/15) com seed. Teste usado **só no final**.

---

## 7. Plano de execução por fases

**Ao final de CADA fase: pare, mostre o que foi feito, rode os testes, explique em 5–10 linhas o que cada parte faz e espere meu "OK" antes de seguir.** Eu preciso entender tudo antes de avançar.

**Fase 0 — Setup**
`requirements.txt` (numpy, matplotlib, scikit-learn, pytest, jupyter), estrutura de pastas vazia, README com o objetivo. Confirme comigo a estrutura.

**Fase 1 — Motor (`motor.py`)**
Tensor, `_como_tensor`, `_desfazer_broadcast`, `+ - * /` (e reversos), `neg`, `@`, `**` (expoente escalar), `exp`, `log`, `sqrt`, `sum`, `mean`, `sigmoid`, `tanh`, `backward()` com ordenação topológica.
→ `test_motor.py`: diferenças finitas centrais (`h = 1e-6`) para **cada** operação, incluindo casos com broadcasting `(3,4)+(1,4)`, `(3,4)*escalar`, nó reutilizado (`x*x`), e sigmoid primitiva × composta.

**Fase 2 — Rede (`rede.py`)**
Inicializações, ativações (ReLU/Leaky por máscara), losses BCE e MSE, `Camada`, `MLP.forward`, `MLP.parametros()`.
→ `test_rede.py`: gradient check da rede inteira para cada ativação.

**Fase 3 — Otimizadores e treino (`otimizadores.py`, `treino.py`)**
SGD, Adam, clipping. Loop com mini-lotes embaralhados, histórico de loss/métrica em treino e validação, coleta para histogramas, early stopping.
→ testes: overfit de mini-lote e equivalência com `np.linalg.lstsq`.

**Fase 4 — Exportar/Importar**
`salvar_modelo` / `carregar_modelo` em JSON.
→ teste de round-trip.

**Fase 5 — Dados e gráficos (`dados.py`, `graficos.py`)**
Carregadores, split, padronização; plots de loss, métrica, histogramas, norma de gradiente por camada, % ReLU mortos.

**Fase 6 — Notebooks de resultados**
`01_regressao`, `02_classificacao` (incluindo comparação sigmoid+BCE × softmax+CE e um pequeno comparativo SGD × Adam e He × Xavier), `03_gradientes` (prever → medir → corrigir). Cada notebook começa com uma célula markdown explicando o que será mostrado e termina com conclusões em 3–5 tópicos. Todos com `MODO_RAPIDO`.

**Fase 7 — Kit de defesa**
`ROTEIRO_DEFESA.md` e `MUDANCAS_PROVAVEIS.md`. **Teste de verdade cada mudança** de `MUDANCAS_PROVAVEIS.md` numa branch/cópia para garantir que o código sugerido funciona, depois desfaça.

**Fase 8 — Revisão de simplicidade**
Releia todo o código procurando: linhas difíceis de explicar, nomes ruins, duplicação, comentários faltando nos `backward`. Simplifique. Rode todos os testes e notebooks do zero (`Restart & Run All`). Liste para mim os 5 trechos mais difíceis do projeto e me explique cada um como se eu fosse apresentar.

---

## 8. Critério de pronto

- `pytest` passa tudo.
- Os 3 notebooks rodam do zero sem erro, em modo rápido < 60 s cada.
- Todos os itens da seção 2 marcados.
- Eu consigo, sem olhar, explicar o `backward` do `@` e o `_desfazer_broadcast`.
