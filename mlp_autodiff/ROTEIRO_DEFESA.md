# Roteiro de defesa

## A ideia em 3 frases

1. Cada operação (`+`, `@`, `exp`...) cria um `Tensor` novo que **lembra quem são seus pais** e guarda uma
   função `_backward` que sabe **repassar o gradiente** para eles (a derivada local × o gradiente que chegou).
2. `backward()` coloca o grafo em **ordem topológica** e chama `_backward` de trás para frente: é a regra da
   cadeia aplicada automaticamente.
3. O MLP, as ativações e as losses são **só composições** dessas operações; nenhuma derivada foi escrita
   fora do `motor.py`.

---

## Arquivo por arquivo

### `motor.py` — o motor de autodiff
| Bloco | O que faz, em linguagem simples |
|---|---|
| `_como_tensor` | Se receber um número ou array, embrulha num `Tensor` sem pais (uma constante). Permite `t + 2`. |
| `_desfazer_broadcast` | Quando o NumPy "esticou" um tensor (ex.: bias `(1,n)` somado a `(batch,n)`), cada elemento original foi usado várias vezes; o gradiente dele é a **soma** das contribuições. A função soma nos eixos esticados para o gradiente voltar ao shape original. |
| `Tensor.__init__` | Guarda `data` (float64), `grad` (zeros, mesmo shape), `parents`, `_backward` (começa sem fazer nada), `nome`, `op`. |
| `__add__`, `__sub__`, `__mul__`, `__truediv__` | Calculam o resultado e definem `_backward` com a derivada de cada lado (comentário em cima). Todas passam por `_desfazer_broadcast`. |
| `__neg__`, `__radd__`... | Negação e versões "reversas" (`2 - t` chama `t.__rsub__(2)`), que só reaproveitam as anteriores. |
| `__matmul__` | `C = A @ B` → `dA = G @ Bᵀ`, `dB = Aᵀ @ G`. Os shapes "fecham" sozinhos (ver pergunta 3). |
| `__pow__`, `exp`, `log`, `sqrt` | Derivadas clássicas: `p·x^(p-1)`, `eˣ`, `1/x`, `1/(2√x)`. `log_base` e `mean` são **compostos** (sem backward próprio). |
| `sum(axis, keepdims)` | A derivada da soma é 1 para cada elemento: o gradiente é "copiado" de volta para todas as posições somadas. |
| `sigmoid`, `tanh` | Primitivas por **estabilidade numérica**: `1/(1+e^-x)` composta dá `inf/inf = NaN` para x muito negativo. Usamos `σ(x) = 0,5·(1 + tanh(x/2))`. O teste mostra que o gradiente é igual ao da versão composta. |
| `backward` | DFS para montar a ordem topológica, `grad` da saída = 1, e chama `_backward` em ordem reversa. |
| `imprimir_grafo` | Imprime a árvore da expressão (para explicar no quadro). |

### `checagem.py` — diferenças finitas
Mexe em **um elemento por vez** (`+h` e `-h`), recalcula a função e usa `(f(x+h) - f(x-h)) / 2h`.
Compara com o `.grad` do motor por **erro relativo** (não depende da escala).

### `rede.py` — rede
| Bloco | O que faz |
|---|---|
| `init_he/xavier/normal/uniforme` | Sorteiam a matriz `W (fan_in, fan_out)`. Bias sempre zero. |
| `relu`, `leaky_relu` | `x * máscara`, com máscara **constante** (0/1 ou α/1). O motor deriva o produto sozinho. |
| `softmax` | Subtrai o máximo da linha (constante), `exp`, divide pela soma. Só composição. |
| `mse`, `bce`, `entropia_cruzada` | Losses compostas. `eps = 1e-7` dentro do `log` evita `log(0)`. |
| `ATIVACOES`, `PERDAS`, `INICIALIZACOES` | Dicionários string → função: config em JSON e fácil de estender. |
| `Camada` | `W`, `b` e `forward: ativacao(x @ W + b)`. |
| `MLP` | Monta as camadas a partir do `config`; `forward` guarda as saídas em `self.ativacoes` (para os histogramas); `parametros()` lista W e b; `prever()` aplica a padronização e devolve na escala original. |

### `exportacao.py` — salvar e carregar
JSON legível com `versao`, `config`, `perda`, pesos (listas), **padronização** e métricas.
`carregar_modelo` cria o `MLP(config)` e sobrescreve os pesos.

### `otimizadores.py`
`Otimizador` (base): guarda parâmetros, `zerar_grad`, clipping. `SGD`: `θ -= η·g`. `Adam`: médias móveis
de `g` e `g²` com correção de viés. `recortar_gradientes`: se a norma global passar do limite, multiplica
todos os gradientes pelo mesmo fator.

### `treino.py`
| Bloco | O que faz |
|---|---|
| `CONFIG_TREINO_PADRAO` | Valores padrão; o usuário só passa o que quer mudar. |
| `r2`, `acuracia`, `metricas_regressao` | Métricas em NumPy puro (não precisam de gradiente). |
| `penalidade_l2` | `Σ W²` composta. |
| `calcular_perda` | forward + loss (+ L2). |
| `coletar_monitoramento` | forward + backward num **lote fixo** e guarda cópias de ativações e gradientes. |
| `gerar_minilotes` | Embaralha e fatia. |
| `treinar` | Por época: para cada mini-lote **zera → forward → backward → passo**; avalia treino e validação; para se divergir; early stopping guardando os melhores pesos. |

### `dados.py`
California (padroniza X e y com estatísticas **só do treino**) e Fashion-MNIST (pixels/255, one-hot,
cache em `.npz`). Split 70/15/15 com seed.

### `graficos.py` e `graficos_gradientes.py`
`graficos.py`: curvas de custo e métrica (treino × validação), comparação entre configurações, matriz de
confusão, real × previsto, tabela final. `graficos_gradientes.py`: histogramas (linhas = épocas, colunas =
camadas; gradientes em log10|g|), norma do gradiente por camada (por época ou por iteração), antes × depois,
sinal e gradiente em função da profundidade, % de ReLU mortos.

---

## As 20 perguntas mais prováveis

**1. Por que ordenação topológica?**
Um nó pode ser usado por vários outros (ex.: `x` em `x*x + exp(x)`). Ele só pode repassar o gradiente
**depois** de receber todas as contribuições. A ordem topológica reversa garante que todos os "filhos" de um
nó já rodaram o `_backward` antes dele. Recursão ingênua (chamar o backward dos pais na hora) repassaria um
gradiente incompleto.

**2. Por que `+=` no gradiente e não `=`?**
Se um tensor é usado em dois lugares, o gradiente total é a **soma** das duas contribuições (regra da cadeia
multivariável). Com `=`, a segunda sobrescreveria a primeira. Exemplo do teste: `d(x·x)/dx = 2x`; com `=`
daria `x`.

**3. Explique o backward do `@`.**
`C = A @ B`, com `A (n,k)`, `B (k,m)`, `C (n,m)`, e `G = dL/dC (n,m)`. Cada `C[i,j] = Σₜ A[i,t]·B[t,j]`, então
`dL/dA[i,t] = Σⱼ G[i,j]·B[t,j]` → `dA = G @ Bᵀ` (shape `(n,m)@(m,k) = (n,k)`). Igual para
`dB = Aᵀ @ G` (`(k,n)@(n,m) = (k,m)`). Dica para lembrar: "o shape tem que fechar com o do original".

**4. Explique o `_desfazer_broadcast`.**
No forward, `X @ W + b` com `b (1,n)` vira `(batch,n)`: o NumPy **repetiu** `b` em todas as linhas. Então cada
`b[j]` influenciou `batch` saídas, e sua derivada é a **soma** dessas `batch` contribuições. A função faz isso:
(1) soma eixos que foram criados à esquerda; (2) soma (mantendo o eixo) os eixos que tinham tamanho 1.

**5. Por que zerar os gradientes a cada passo?**
Porque o motor acumula (`+=`). Sem zerar, o gradiente do passo 2 seria somado ao do passo 1 e a atualização
ficaria errada (cada vez maior).

**6. Por que ReLU não tem backward próprio?**
`relu(x) = x * m`, com `m = (x > 0)` como constante. A derivada do produto em relação a `x` é `m`: exatamente
a derivada da ReLU. O motor faz isso sozinho com o `__mul__`.

**7. Então por que sigmoid e tanh são primitivas?**
Não é por necessidade matemática — a versão composta `1/(1+exp(-x))` dá o mesmo gradiente (há teste
provando). É por **estabilidade**: para x = -1000, `exp(1000) = inf` e o backward vira `inf/inf = NaN`,
justamente no experimento de explosão.

**8. Por que He para ReLU e Xavier para Tanh/Sigmoid?**
Queremos que a **variância** do sinal fique constante de camada em camada. Para uma camada linear,
`Var(saída) = fan_in · Var(W) · Var(entrada)` → `Var(W) = 1/fan_in`. A ReLU zera metade das entradas
(corta a variância pela metade), então He usa `2/fan_in`. Xavier faz a média entre forward (`fan_in`) e
backward (`fan_out`): `2/(fan_in+fan_out)`, e foi derivado para ativações simétricas perto de 0 (tanh).

**9. O que é a correção de viés do Adam?**
`m` e `v` começam em 0. Nos primeiros passos, `m = (1-β₁)·g` é bem menor que `g` (ex.: 0,1·g). Dividir por
`(1-β₁ᵗ)` corrige isso: em `t=1`, `m̂ = 0,1g / 0,1 = g`. Com `t` grande, `βᵗ → 0` e a correção some.

**10. Por que validação separada do teste?**
A validação é usada para **decisões** (early stopping, escolher hiperparâmetros), então ela "vaza" para o
modelo. O teste só é usado **uma vez no final** para estimar o desempenho em dados realmente novos.

**11. Por que padronizar só com estatísticas do treino?**
Usar média/desvio de validação ou teste seria espiar dados que, na vida real, não teríamos. Por isso as
mesmas estatísticas do treino vão no arquivo exportado.

**12. Por que a padronização vai no arquivo exportado?**
A rede aprendeu com entradas padronizadas. Se alguém importar o modelo e passar dados brutos, o resultado
fica errado. `prever()` aplica a padronização da entrada e desfaz a do alvo.

**13. Como funciona BCE com 10 classes?**
BCE é binária. Usamos one-hot e **uma sigmoid por saída**: cada saída responde "é da classe k?". São 10
problemas binários independentes. A classe prevista é o `argmax`. Diferente da softmax, as probabilidades
não somam 1.

**14. Por que `eps` dentro do `log`?**
Se `p = 0` (ou `1 - p = 0`), `log(0) = -inf` e o gradiente `1/p` explode. Com `log(p + 1e-7)` o valor fica
finito. O `eps` é composto com `+` do motor, sem tratamento especial.

**15. Por que subtrair o máximo na softmax?**
`softmax(x) = softmax(x - c)` para qualquer constante `c` (o `e^-c` cancela). Com `c = max`, o maior
expoente é `e⁰ = 1` e nada estoura.

**16. Por que a dissipação acontece com sigmoid?**
O gradiente da 1ª camada é um produto de L termos `σ'(z)·W`. Como `σ'(z) ≤ 0,25`, só a ativação já dá um
fator ≤ 0,25ᴸ (≈ 10⁻⁶ para L = 10). Medimos 1,5·10⁻⁸ (os pesos pequenos encolhem mais ainda).

**17. Por que a explosão com Normal(1.0)?**
Com 32 entradas e ReLU, cada camada multiplica a variância por `32 · 1² / 2 = 16`, ou seja, o desvio por
`4`. Em 10 camadas, `4¹⁰ ≈ 10⁶`. Medimos 4,3× por camada. O custo começa em 10¹² e vira NaN na 3ª iteração.

**18. O que o gradient clipping faz e por que não resolve sozinho?**
Se `‖g‖ > limite`, multiplica todos os gradientes por `limite/‖g‖`: mantém a direção e limita o tamanho do
passo. Evita o NaN, mas com a init ruim a rede continua num ponto péssimo (R² ≈ -10¹¹). Precisa da init correta.

**19. Por que diferenças finitas centrais e não `(f(x+h)-f(x))/h`?**
O erro da central é O(h²); o da progressiva é O(h). Com `h = 1e-6` em float64, a central dá erro ~1e-10.
Por isso usamos `float64` em tudo.

**20. Por que o gradient check falha na ReLU em x = 0?**
A ReLU tem uma "quina" em 0: a derivada pela esquerda é 0 e pela direita é 1. Se `x` estiver a menos de
`h` de 0, `x+h` e `x-h` caem em lados diferentes e a diferença finita dá ~0,5. Nos testes, entradas contínuas
aleatórias tornam isso desprezível.

### Bônus

**21. Como você sabe que tudo junto está certo?** Quatro provas ponta a ponta em `test_rede.py`: gradient
check da rede inteira; **equivalência com mínimos quadrados** (MLP sem camada oculta + linear + MSE treinado
com Adam chega na solução de `np.linalg.lstsq`); **overfit** de 16 exemplos (loss → 0); round-trip da exportação.

**22. Por que não Boston Housing?** Foi removido do scikit-learn 1.2: a variável `B` foi construída a partir da
proporção de moradores negros com uma suposição racista embutida. Usamos California Housing.

**23. Por que os gradientes das ativações saem "de graça"?** Porque cada ativação é um `Tensor` do grafo;
depois do `backward()` o `.grad` dele já é `dL/d(ativação)`. Basta ter guardado a referência (`rede.ativacoes`).

**24. Por que histograma num lote fixo?** Se o lote mudasse a cada coleta, a diferença entre histogramas
misturaria "a rede mudou" com "os dados mudaram". Lote fixo isola o efeito do treino.
