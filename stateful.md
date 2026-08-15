# Um cache de prefixo: span, imutável, para todos os modelos

Plano para a codebase ter **uma** maneira de reaproveitar prefill: `resume(tokens, cache)`
antes, `commit(tokens, cache)` durante e depois, sempre sobre spans imutáveis endereçados
por conteúdo, sempre com VRAM e SSD atrás da mesma abstração. Sem trie por token, sem
posse, sem `Spill`, sem caminho por família, sem "este modelo não rebobina".

## O que esta revisão mudou

Correções materiais sobre a versão anterior, conferidas contra o código e os checkpoints:

- **Retomada de híbrido precisa de âncora, e a retenção anterior não a garantia.** O
  adaptador Anthropic descarta blocos de thinking reenviados (`server/anthropic.py:161` —
  "read and dropped") e um tool call re-serializado não reproduz os ids gerados, então o
  turno seguinte **não** estende os ids do decode: estende o prompt do turno anterior. Um
  "tip" único no fim do decode deixaria o híbrido com cobertura zero no caso comum. A
  retenção agora é **duas âncoras por conversa**: a última fronteira do prefill e a última
  fronteira cruzada no decode. `snapshots_per_chain` saiu; divergência no meio do
  histórico num híbrido é reprefill, registrado como risco.
- **O prefill tem que parar na última fronteira.** O laço de blocos alimenta o último
  bloco parcial inteiro (`core/prefill.py`, e `BatchPrefill.advance`), então as fronteiras
  internas nunca são "pisadas" e um replay com `max_tokens=1` não deixaria âncora nenhuma.
  O último bloco passa a ser cortado na última fronteira `<= N-1`. Sem isso, "ganho no
  segundo turno" para híbrido era falso no próprio protocolo de validação do §11.
- **Só `Snapshot` exige fronteira; `Rows` não.** Linhas são fatias do buffer vivo e podem
  ser cortadas em qualquer momento posterior. A rodada especulativa só precisa de capping
  por causa das camadas `Snapshot`, e a fórmula anterior tinha um furo em `room == 0`
  (cruzava a fronteira dentro do forward com largura cheia). Regra corrigida no §2.6.
- **Ring reclassificado.** Nenhum `make_cache` da árvore devolve `RingKVCache` — laguna e
  lfm2 devolvem `KVCache` e o ring só existe como forma de decode pós-promoção
  (`laguna/model.py:91-98`, `lfm2/dense/model.py:74`). Camada sliding é `Rows()` (o cache
  vivo do prefill guarda o histórico inteiro, mascarado); a restrição `span <= janela`
  vale só para o commit de spans fechados **durante o decode** promovido. `keep` virou
  otimização de leitura (zero-fill exato sob a máscara de janela), não de armazenamento —
  a regra anterior "span <= keep senão Snapshot" era insuficiente de qualquer forma: com
  bloco de prefill 2048 > janela, as linhas dos spans internos morreriam antes do commit.
- **Números recalculados dos configs.** Nemotron 3.5 Lightning tem **23** camadas mamba
  (não ~40; 52 = 23 mamba + 6 attention + 23 MoE sem estado), snapshot **49 MB** (não
  ~80), KV **6 KB/token** (não 310 KB/token de snapshot). Qwen3.8-27B: 151 MB de estado
  + ~3 MB de janelas conv = **154 MB**; KV 64 KB/token. Laguna XS: 40 camadas (10 full +
  30 sliding, janela 512), **160 KB/token** de linhas — e o custo de guardar as sliding
  como `Rows` é 3x o útil, quantificado no §6. Capturar snapshot em memória é reter uma
  referência (~0, não ~250 µs); o custo de cópia existe só na materialização para disco.
- **O caminho principal é o batched.** 45/45 famílias têm `continuous_batching = True`;
  `stream_ids` serve só prompts Iterator e imagens. A integração primária é
  `language.begin_batch` / `batching.BatchPrefill` / `batching.step` / `TextBatch.finish`
  — `generate.stream_ids` é o secundário. A versão anterior mirava o caminho single.
- **Tabela por camada corrigida.** `NgramCache` e `MLACache` são folhas, não compostos:
  `MLACache` é `Rows()` (dois buffers crescidos no eixo 2, espelho de `KVCache`);
  `NgramCache` é `Snapshot` (os últimos `n-1` ids). `Composite` cobre **duas** classes
  (`DSACache`, `DeepseekV4Cache`), não quatro. `LatentKVCache` herda de `KVCache` e não
  custa nada. A base `LayerCache` ganha `stored()`/`restore` triviais — as camadas MoE
  sem estado do nemotron_h são hoje o que torna o trunk inteiro não-storable.
- **A precondição de RoPE foi reverificada família a família**, não por uma linha de
  `rope.py`: `llama`/`exaone4` recusam tudo que não é `llama3` por nome; o "dynamic" do
  hunyuan_dense é rescaling estático; o longrope do phi3 fixa a tabela `long_factor` na
  construção. Nenhuma família registrada monta tabela em função do comprimento corrente.
- **Caminho frio reescrito** (§6): write-behind idempotente para spans, âncora filada só
  em drain/eviction, formato por arquivo, contagem de arquivos, leitura estimada vs
  reprefill por família, e o fato de que uma conversa laguna de 62k **não cabe** no teto
  default de disco. Restart limpo é garantido; crash perde a âncora dos híbridos — dito.
- **Nomes reais de config**: `prefix_cache_bytes` e `prefix_disk_bytes` (não
  `prefix_memory_bytes`); `DELETE /admin/prefixes/{tier}` já existe (`server/state.py:145`).
  O piso de spill (`_SPILL_FLOOR`) morre — span tem tamanho natural.
- **Multimodal fica fora.** A identidade do span são os ids; um prompt com imagem tem KV
  que depende dos pixels. O caminho de visão não usa prefixo hoje e continua sem usar.
- **Escopo da medição**: `omnia-bench` mede decode com prefixo desligado (default 0). Um
  ganho de bench vindo de hit de prefixo é exatamente o artefato que a skill
  `measurement` rejeita (cache chaveado no input do request cujo único hit é o bench
  repetindo o prompt). O TTFT do prefixo é medido pelo replay do §11; o decode não pode
  regredir, e isso sim é `omnia-bench paired`.
- Citações menores: o check `offset == len(committed) - 1` está em
  `stream_speculative_ids` (`speculative.py:565`), não em `_compiled_round`; a recusa
  atual descarta recorrente **só quando a entrada é mais longa que o match** — prefixo
  exato já funciona hoje, uma vez, destrutivamente (`prompt_cache.py:258-269`).

---

## 1. As N maneiras de hoje (o que este plano colapsa)

1. **Trie por token com posse** (`core/prompt_cache.py`) — `take` entrega o cache e o
   **remove** da trie (`_drop(best)` dentro de `take`). Duas requisições concorrentes com
   o mesmo system prompt não compartilham nada; a segunda reprefilla inteiro.
2. **Rebobinagem** (`is_trimmable`, `prompt_cache.py:258`) — uma entrada mais longa que o
   match é rebobinada, ou descartada inteira quando alguma camada não rebobina. Para
   recorrente e ring o reuso só existe no prefixo exato; qualquer divergência é descarte.
3. **Snapshot manual, um só, e só no caminho single** (`generate.py:371` `_boundary`) —
   a fronteira prompt→geração para trunks que não rebobinam. O caminho batched — que é o
   que o servidor usa — não tem equivalente: `TextBatch.finish` (`language.py:340`)
   insere só a forma final, e reporta `kept_prefix` como `prefix is not None`
   (`language.py:364`), sem as três condições do single.
4. **Promoção vs. trie** (`generate.py:619-630`) — compilar o decode é condicionado a a
   trie ter onde ficar de pé; `_grown` (`language.py:317`) desfaz promoção para inserir.
5. **Papéis** (`_EVICTION_ORDER`, `system` > `user` > `assistant`) — proxy de
   compartilhamento, com dois `insert` por requisição no single e contagem dupla de bytes.
6. **Teto por trie + `Ledger` + weakrefs** (`Budget`) — vítima comparada entre tries de
   classes de cache diferentes, porque cada modelo tem a sua e ela morre com o modelo.
7. **Tier de disco à parte** (`server/prefixes.py`) — protocolo `Spill`, chave por
   conversa inteira, escrita só na eviction, e um `recall` que **varre todos os arquivos**
   do modelo calculando prefixo comum id a id.
8. **Cinco classes sem `stored`/`restore`** — `PoolCache`, `DeepseekV4Cache`, `DSACache`,
   `NgramCache`, `MLACache` (deepseek_v4, glm4_moe-DSA, longcat). E as camadas MoE do
   nemotron_h são `LayerCache` puro, cuja base não responde `stored` — o que torna o
   trunk inteiro não-storable e tira o Lightning do disco hoje.
9. **Especulação excluída** (`generate.py:471` levanta; `language.py:505-510` desliga o
   prefixo em silêncio quando há proposer, no caminho batched).

---

## 2. A forma final

### 2.1 O span

Unidade única de retomada: `span` tokens, imutável, chave encadeada.

```
tokens:  [--- span 0 ---][--- span 1 ---][--- span 2 ---][ cauda parcial ]
chaves:        k0              k1              k2          (não guardada)

k_i = H(k_{i-1} ‖ ids ‖ model ‖ stamp ‖ policy ‖ span ‖ layout_version)
```

Cai fora daqui, sem código adicional:

- **Reuso é o maior prefixo de chaves presente.** Span 1 diverge → 0 é aproveitado,
  1..n são prefillados. É "múltiplos prefixos", e é só o encadeamento.
- **Nada é entregue e nada é rebobinado.** Adotar é copiar linhas para o buffer contíguo
  do pedido. Duas requisições adotam os mesmos spans ao mesmo tempo.
- **Um span fechado nunca muda.** É o que faz o disco ser write-once idempotente (dois
  daemons ou duas requisições que fecham o mesmo span escrevem os mesmos bytes sob a
  mesma chave) e o que dispensa coordenação entre requisições concorrentes.
- **A cauda parcial não é guardada.** Custo: `<= span - 1` tokens reprefillados por
  turno. Em troca some todo o caso especial de "último bloco". Um token sempre sobra para
  o forward que produz os logits, então a cobertura máxima é a maior fronteira
  `<= len(tokens) - 1`.

Duas pré-condições, verificadas e não supostas:

- **As linhas na posição `j` dependem de `tokens[:j+1]` e de mais nada.** As tabelas de
  RoPE são estáticas por config em toda família registrada: `core/rope.py` constrói
  `llama3` e YaRN a partir do config; `llama/config.py:17` e `exaone4/config.py:11`
  recusam `linear`/`dynamic`/`yarn` por nome; o `rope_type: "dynamic"` do hunyuan_dense
  é um rescaling NTK-alpha **estático** (`hunyuan_dense/config.py:52`); o longrope do
  phi3 fixa a tabela `long_factor` na construção (`phi3/config.py:27`). Um RoPE futuro
  que escale com o comprimento corrente teria de entrar na política da chave — e hoje
  nenhum escala.
- **A identidade do prompt são os ids.** Um prompt com imagem tem KV que depende dos
  pixels e não entra no store — o caminho de visão não usa prefixo hoje e o plano não
  muda isso. Template, effort e tools participam da chave do único jeito honesto: mudam o
  render, o render muda os ids.

### 2.2 O contrato por camada

Cada camada declara **como cada tensor de `stored()` compõe ao longo dos spans**. Duas
formas, e só duas:

```python
@dataclass(frozen=True)
class Rows:
    """Histórico: o span guarda as linhas que os tokens dele produziram.

    `stride` é tokens por linha (1 = KV; `ratio` no pool comprimido do DeepSeek).
    `keep` é o sufixo que basta para retomar (janela sliding): spans mais fundos que
    `keep` não são lidos e viram zeros — exato porque a máscara de janela zera o peso
    dessas linhas antes de qualquer leitura (§3)."""

    axis: int = 2
    stride: int = 1
    keep: int | None = None


@dataclass(frozen=True)
class Snapshot:
    """Estado: o valor inteiro na fronteira. Retomar é ler a âncora; não compõe."""


type Layout = Rows | Snapshot
```

`stored()` e `restore()` já existem na base e em todas as classes do core; faltam nas
cinco do §1.8 e na própria base (`stored() -> {}` e `restore` que assenta o offset — o
que apaga a razão de o nemotron_h ser não-storable hoje). `is_storable` some: declarar
layout é o que substitui, e `{}` é "participo só pelo offset" (`SharedKVReader`, camadas
MoE sem estado).

O layout default é da classe; o trunk pode sobrepor por índice de camada, porque a classe
nem sempre sabe: as camadas sliding do laguna são `KVCache` comuns na construção — a
janela é `config.layer_types` + `config.sliding_window`, conhecimento da família. Um
protocolo pequeno, resolvido **uma vez** na construção do pedido contra o span, na forma
de `core/kernels/resolve.py`:

```python
@runtime_checkable
class Layouts(Protocol):
    def cache_layouts(self) -> Sequence[Mapping[str, Layout]] | None: ...
```

O núcleo do `resume`:

```python
keys = chain.keys(tokens, self.span)                 # até a maior fronteira <= len-1
reach = sum(1 for _ in takewhile(self._present, keys))
anchor = reach * self.span
if self._has_snapshots:
    anchor = self._deepest_anchor(keys[:reach])      # fronteira com âncora, <= reach
if anchor == 0:
    return 0
spans = keys[: anchor // self.span]
payloads = [self._fetch(key) for key in spans]
snapshot = self._snapshot_at(spans[-1]) if self._has_snapshots else {}
for index, layer in enumerate(into):
    tensors: dict[str, mx.array] = {}
    for name, layout in self._resolved[index].items():
        match layout:
            case Rows(axis=axis, keep=keep):
                parts = self._parts(payloads, f"{index}.{name}", keep)
                tensors[name] = mx.concatenate(parts, axis=axis)
            case Snapshot():
                tensors[name] = snapshot[f"{index}.{name}"]
            case _:
                assert_never(layout)
    layer.restore(anchor, tensors)
return anchor
```

Três fatos que o esboço carrega:

- **Todas as camadas assentam no mesmo `anchor`.** Um trunk com camadas `Snapshot` só
  retoma até uma fronteira que tenha âncora; se as linhas alcançam mais fundo que a
  âncora, o excedente é reprefillado. Trunk sem `Snapshot` retoma até `reach * span`.
- **`_parts` com `keep`**: os spans que não alcançam `[anchor - keep, anchor)` entram
  como zeros do shape certo, sem leitura. Exato sob a máscara de janela; §3 e mutação.
- `restore(anchor, {})` numa camada de layout vazio assenta só o offset — o invariante 1
  do §8 é atendido por construção.

`commit` é o espelho: corta `stored()` por span, `mx.contiguous` (uma view retida segura
o buffer inteiro do pedido vivo — e no decode compilado impediria a doação de buffer do
grafo), `mx.eval` na thread que gerou, e entrega ao store. Corta até
`min(len(tokens), rows)` — **`rows`, não `offset`**: no caminho compilado a posição vive
no grafo e `offset` é o valor que o trace congelou; é a distinção que `LayerCache.rows`
já faz e que `cache_file.dump` já respeita.

### 2.3 Fronteiras: onde `Rows` fecha e onde `Snapshot` é capturado

Os dois layouts têm cadências diferentes, e confundi-las era o erro da versão anterior:

- **`Rows` fecha span em qualquer momento posterior.** As linhas estão no buffer vivo (ou
  no buffer fixo do decode compilado, lidas por `rows`); o commit fatia quando quiser. A
  cadência escolhida: na travessia de cada fronteira (prefill e decode), para não segurar
  buffers promovidos vivos por referência até o fim da geração.
- **`Snapshot` exige a camada parada na fronteira.** Estado recorrente em posição
  intermediária de um forward não existe fora do kernel. Capturar é **reter a
  referência** do tensor de estado daquele passo (DeltaCache/Mamba reatribuem por passo;
  no decode compilado a referência retida impede a doação do buffer) — custo ~0 em
  memória; a cópia só existe quando a âncora vai para o disco.

Onde o trunk fica parado numa fronteira:

1. **Entre blocos de prefill** — `BLOCK = 2048` é múltiplo do span, então toda borda de
   bloco é fronteira. Já são os pontos de `mx.eval` das caches.
2. **No fim do prefill, por construção**: o último bloco parcial é cortado na última
   fronteira `<= N-1` (um forward a mais, custo de um split). É a **âncora de prefill** —
   o ponto que o turno seguinte estende, seja qual for o adaptador. Sem esse corte, um
   turno de `max_tokens=1` não deixa âncora nenhuma e o híbrido não ganha no 2º turno.
3. **A cada passo do decode** — o laço é por token em Python (single, batched e
   compilado), então toda travessia `rows % span == 0` é um ponto parado. A detecção é
   aritmética sobre o comprimento assentado; nenhum sync novo.
4. **Entre rodadas especulativas** — com o capping do §2.6.

Retenção de snapshot, por conversa (cadeia):

- **Âncora de prefill**: a última fronteira do prompt. Substituída quando o turno
  seguinte prefilla mais fundo na mesma cadeia (supersessão no commit, que conhece a
  lista de chaves ancestrais).
- **Âncora de decode**: a última fronteira cruzada no decode, substituída a cada
  travessia. Serve o cliente que ecoa os ids gerados literalmente (completions cru,
  replay teacher-forced); o Anthropic descarta thinking e re-serializa tool calls, então
  para ele quem trabalha é a âncora de prefill.
- Um snapshot superado é **largado, nunca filado**: escrever 58 estados de 154 MB num
  prefill frio de 15k era o defeito que a supersessão existe para impedir.

Custo residente por conversa ativa num híbrido: até duas âncoras — Qwen3.8-27B 2×154 MB,
Lightning 2×49 MB, laguna 0. Sob pressão a âncora de decode cai primeiro.

### 2.4 O store: uma abstração, duas residências

```python
@dataclass(frozen=True)
class Held:
    payload: Payload          # tensores nomeados + os ids do span
    nbytes: int
    filed: bool               # despejar custa uma escrita, ou não


@dataclass(frozen=True)
class Filed:
    nbytes: int


type Residency = Held | Filed


@runtime_checkable
class Vault(Protocol):
    """O andar frio, injetado pelo daemon. Não sabe o que é um trunk."""

    def holds(self, key: Key) -> bool: ...
    def read(self, key: Key) -> Payload | None: ...
    def write(self, key: Key, payload: Payload, nbytes: int) -> None: ...
    def forget(self, key: Key) -> None: ...
```

- **Spans (`Rows`) são filados por write-behind**: uma fila de profundidade limitada,
  fora da thread do request, na cadência em que fecham. Content-addressed e write-once,
  a escrita é idempotente — dois writers do mesmo span são um no-op. Um span já filado é
  despejado da VRAM de graça.
- **Âncoras (`Snapshot`) não entram no write-behind.** Numa conversa monotônica a âncora
  é superada a cada turno; escrevê-la toda vez seria ~154 MB por turno de SSD para um
  arquivo que morre no turno seguinte. Ela vai ao disco em dois eventos: **drain** (o
  shutdown limpo que `Budget.drain` já faz hoje) e **eviction por pressão**. Consequência
  dita no §9: restart limpo preserva híbridos; crash preserva só as linhas.
- `_fetch` procura em memória, cai no vault, e **promove o que leu**: VRAM é hot path por
  consequência do LRU, não por decisão do chamador — que nunca sabe de onde veio. O meter
  sabe (§11): esconder a residência do chamador não é escondê-la da medição.
- **Um store por daemon, não por modelo**: a chave carrega checkpoint e stamp, então um
  teto de VRAM e um de disco bastam. Isso apaga `Budget`, `Ledger`, `victim()` e os
  weakrefs — e dá de brinde cache quente sobrevivendo a unload/reload do modelo, porque o
  store não morre mais com a facade. Também apaga o `Spill[Any]` de `GenerationOptions`:
  o store trafica `dict[str, mx.array]` e o muro de tipo-de-elemento deixa de existir.

### 2.5 O fluxo

O caminho batched é o principal (45/45 famílias; o servidor só usa ele para prompt
inteiro), o single fica para prompts Iterator e imagens:

```python
covered = store.resume(encoded, cache, chain=chain)   # language.begin_batch
# BatchPrefill(reused=covered, ...) — commit entre blocos + split da última fronteira
# batching.step / _shared_step — commit na travessia, por sequência
# TextBatch.finish — fecha os spans restantes; some o _grown/StandingLayer
```

e, no single, `stream_ids` chama o mesmo par nos mesmos pontos (pré-prefill, entre
blocos, travessia no decode, fim). Os dois caminhos entram **juntos** na fase 1 — senão
convivem dois mecanismos.

### 2.6 Especulação

Deixa de ser incompatível. O invariante que a especulação já mantém é o que `commit`
precisa: **entre rodadas, as linhas `[0, rows)` são os ids assentados `[0, rows)`** — uma
rodada aceita tudo e não desfaz nada, ou volta por `trim`/replay/rewind ao comprimento
assentado, nunca à frente (`stream_speculative_ids` chega a checar
`target_cache[0].offset == len(committed) - 1` antes da rodada compilada,
`speculative.py:565`).

Duas regras:

1. **Nada de `commit` dentro da rodada.** O forward de verificação escreve `width + 1`
   linhas antes de saber quantas sobrevivem; guardá-las sob a chave dos ids assentados é
   a pior classe de bug daqui. `commit` recusa quando `rows > len(tokens)`.
2. **Fronteira de `Snapshot` não é pulada dentro de um forward** — só afeta trunk com
   camada `Snapshot`, que é exatamente o alvo dos MTPs (nemotron_h, qwen3_5). O draft é
   **truncado** antes do verify:

   ```python
   room = boundary - len(committed)          # committed = offset + 1 entre rodadas
   if 0 < room < width:
       drafted = drafted[:room]              # aceitação cheia para na fronteira
   # room == 0: a rodada cruza; esta fronteira fica sem âncora e o tip fica
   # um span atrás — cobre só o cliente que ecoa ids, e no máximo 1 span.
   ```

   Aceitação parcial para aquém da fronteira e a rodada seguinte tenta de novo. Custo:
   uma rodada encurtada por span (~85 rodadas por span com largura 4), e o caso
   `room == 0` (~1/span das rodadas) perde uma âncora de decode — a âncora de prefill não
   é afetada. O prefill não é especulativo, então as fronteiras do prompt são todas
   cruzadas paradas, de graça.

Por proposer, contra o código:

| proposer | estado próprio | com span |
|---|---|---|
| `Chained` (MTP) | nenhum entre rodadas (`make_cache()` por proposta) | compatível já: `absorb` guarda só a última linha, e ela é a última posição do último bloco do prefill retomado |
| `Autoregressive` | KV próprio, exige trimável | compatível: a rodada 1 já alimenta `committed[offset:]`, então o draft reprefilla o prompt por conta própria — fora do TTFT (acontece depois do 1º token); cadeia própria do draft é otimização futura |
| `Persistent` (MTP com histórico) | um par por posição, escrito das features do alvo | **exige** cadeia própria: com o alvo retomado, as features das posições retomadas não existem e o assert de `propose` (`speculative.py:466-469`) dispara — corretamente. A saída é retomar o cache da cabeça pela sua própria chain (alvo + cabeça + stamps; uma camada contra as 6-16 do alvo), e `absorb` recebe só a cauda |

Enquanto a chain do `Persistent` não existir, MTP-persistente + prefixo continua recusado
e o assert é o guarda. `Chained` e `Autoregressive` destravam na fase 5; o `None` de
`language.py:505-510` e a recusa de `generate.py:471` saem nessa fase.

---

## 3. Por camada: o que cada classe precisa

| camada | layout | trabalho |
|---|---|---|
| `KVCache`, `LatentKVCache` (herda), `FixedKVCache` | `Rows()` | declarar; `FixedKVCache.stored` já corta por `rows` |
| `QuantizedKVCache` | `Rows()` por tensor (codes/scales/biases e a cabeça densa são linhas no eixo 2; contagens por tensor variam por span e o concat não se importa) | declarar; `signature` (formatos, `start_tokens`) segue para a política da chave |
| `LayerCache` base (MoE do nemotron_h, camadas sem estado) | `{}` | `stored() -> {}` e `restore` que assenta offset, na base |
| `SharedKVReader` (gemma3n, gemma4) | `{}` | nenhum: participa pela camada que armazena |
| Sliding como `KVCache` + máscara (laguna, gpt_oss, lfm2 attention) | `Rows(keep=janela)` via `Layouts` do trunk | commit lê do ring em ordem absoluta quando a camada foi promovida (span `<= janela`, checado na resolução); resume pode zero-fill além de `keep` — exato porque a máscara `columns > offset - window` zera o peso e `RingKVCache.promote` copia só `[offset-window, offset)` |
| `RingKVCache` | não é declarada: é forma de decode. `stored()` passa a devolver ordem absoluta (últimas `min(offset, window)` linhas; a rotação sai do formato e o `signature` de rotação morre); `restore` re-rotaciona em duas cópias; vetoriza o laço O(window) de `promote()` de quebra | se um `make_cache` futuro devolver ring, a camada resolve para `Snapshot` (janela em ordem absoluta na âncora) — o prefill em blocos maiores que a janela destrói as linhas antes do commit |
| `ConvCache`, `DeltaCache`, `FixedDeltaCache` (herda) | `Snapshot()` | declarar. Estado não se recomputa sem um forward do gap: a entrada da camada é a saída do bloco anterior |
| `MLACache` (longcat) | `Rows()` | `stored`/`restore` novos, espelho de `KVCache` (dois buffers no eixo 2) |
| `NgramCache` (longcat) | `Snapshot()` | `stored`/`restore` novos: o contexto são os últimos `n-1` ids, dezenas de bytes |
| `PoolCache` (deepseek_v4) | `pooled: Rows(stride=ratio)`; `previous` (overlap): `Snapshot()` | `stored`/`restore` novos. Com `ratio | span`, `remainder == 0` em toda fronteira e a cauda não é estado; `pooled_rows` é `anchor // ratio`, recomputado e não gravado |
| `DSACache`, `DeepseekV4Cache` | delegado | base `Composite` em `core/cache.py` com sub-caches nomeados, delegando `offset`, `nbytes`, `tensors`, `stored`, `restore`, `layout`. As duas classes escrevem essa delegação à mão hoje |

Os ratios do deepseek_v4 são `{0 (LOCAL), 4 (overlap), 128}` (`deepseek_v4/config.py`):
o span efetivo tem de ser múltiplo de 128, e 256 é. `Snapshot` no `previous` e no
`NgramCache` faz dessas famílias "híbridas" para efeito de âncora — âncoras de bytes
pequenos, mesma máquina.

---

## 4. Configuração

```python
prefix_cache_bytes: int      # teto de VRAM (existe; default max(1 GiB, memory_limit/32))
prefix_disk_bytes: int       # teto de SSD (existe; default 8 GiB; 0 = só memória)
prefix_span: int = 256       # granularidade; múltiplo de 64, 64..4096, igual para todo trunk
```

O span efetivo por modelo é arredondado para cima ao múltiplo do maior `stride` das suas
camadas (deepseek_v4: 128), e entra na chave — `prefix_span = 192` não quebra nada, só
não casa com o que 256 gravou. PATCH em runtime é permitido: nada corrompe, os spans
antigos param de casar e caem no LRU por serem, por definição, os menos usados. O que não
pode é trocar o span com uma requisição no voo — a chain é capturada no início do pedido.

Os dois lados da granularidade são reais: para baixo, retomar 62k a span 64 monta ~116
mil arrays em Python por requisição (968 partes × 60 camadas × 2 tensores); para cima, a
cauda parcial nunca guardada vira TTFT jogado fora em todo turno (~1 s de prefill a 4096).
256 é onde os dois somem.

`snapshots_per_chain` não existe: as duas âncoras do §2.3 são estruturais, não knob.
Divergência no meio do histórico num híbrido é reprefill (§9); comprar pontos
intermediários de retomada é trabalho futuro se algum tráfego real precisar.

`_SPILL_FLOOR` morre: o piso existia porque a unidade era a conversa inteira e uma curta
não pagava o arquivo; um span tem tamanho fixo por família e ou o teto o comporta ou o
LRU o come.

---

## 5. Os números por família (calculados dos configs e headers; a medir na fase 2)

Os três modelos de validação do §11, com os shapes lidos dos configs dos checkpoints
locais (`local/Qwen3.8-27B-nvfp4`, `mlx-community/Nemotron-3.5-Lightning-30B-A3B-nvfp4`,
`poolside/Laguna-XS-2.1-NVFP4-mlx`):

| | Qwen3.8-27B | Nemotron 3.5 Lightning | Laguna XS 2.1 |
|---|---|---|---|
| camadas | 48 DeltaNet + 16 attention | 23 mamba + 6 attention + 23 MoE `{}` | 10 full + 30 sliding (janela 512) |
| âncora (`Snapshot`) | 48×[1,48,128,128] fp32 = 151 MB + ~3 MB conv ≈ **154 MB** | 23×[1,64,64,128] fp32 = 48 MB + ~1 MB conv ≈ **49 MB** | **0** |
| linhas (`Rows`) | 16×2×4×256×2 B = **64 KB/token** | 6×2×2×128×2 B = **6 KB/token** | full 40 KB/token + sliding 120 KB/token = **160 KB/token** |
| span de 256 | 16,8 MB | 1,6 MB | 41,9 MB (31,5 MB de sliding, úteis só a `keep` do tip) |
| conversa de 15k | 0,98 GB + âncoras | 92 MB + âncoras | 2,5 GB |
| conversa de 62k | 4,1 GB + âncoras | 380 MB + âncoras | 10,1 GB |

Custos de captura, na banda de cópia (610 GB/s):

- fechar um span (`mx.contiguous` das fatias): Qwen ~28 µs, Lightning ~3 µs, laguna
  ~69 µs — um prefill de 62k soma 7-17 ms de cópia, < 1% do prefill;
- capturar âncora em memória: reter referências, ~0;
- materializar âncora para disco (drain/eviction): 154 MB a ~5 GB/s de escrita ≈ 31 ms.

Duas consequências que a tabela não deixa esconder:

- **As sliding do laguna como `Rows()` custam 3x o útil.** O cache vivo do prefill já
  guarda o histórico inteiro dessas camadas (máscara, sem eviction — `laguna.md`), então
  a retomada não gasta mais VRAM que o prefill frio gastaria; mas o *store* carrega 120
  KB/token dos quais só a janela serve à retomada. `keep` corta a **leitura** (§6), não o
  armazenamento — poda por camada de spans imutáveis compartilhados exigiria refcount por
  cadeia e fica explicitamente fora do escopo, registrada como risco (§9).
- **62k de laguna não cabem no default de 8 GiB de disco.** Cabe até ~48k; acima disso o
  LRU come o começo da conversa e a cobertura encolhe. O protocolo do §11 registra teto e
  ocupação por arm.

---

## 6. O caminho frio

**Formato.** Um arquivo safetensors por payload, em
`~/.cache/mlx_omnia/prefixes/<model-slug>/<digest>.safetensors`, diretório `0700`, staging
+ rename atômico (a forma de `cache_file.dump` hoje). O payload de um span leva os
tensores `"{layer}.{name}"` daquele span e os **ids do span** (~1 KB contra MBs);
o de uma âncora leva os tensores `Snapshot` e a fronteira. Offsets e span ficam no
`__meta__`. Leitura termina em `mx.eval` (trap 2 da casa: mmap lazy corrompe a primeira
avaliação).

**Índice.** A tabela `prefix_cache` do `server/store.py` encolhe: `key`, `model`, `kind`
(span | âncora), `path`, `bytes`, `created_at`, `used_at`. Somem `ids` e `tokens` —
busca é lookup por chave encadeada, nunca mais varredura computando prefixo comum. Os
ids conferidos na leitura vivem no payload (invariante 3 do §8). `sweep` no boot continua
(linha sem arquivo morre); arquivo sem linha morre com o diretório do modelo, como hoje.

**Escrita.** Write-behind para spans (fila limitada, thread própria, serializada —
`DiskSpill._write` já tem a forma); âncora só em drain e eviction (§2.4). O `mx.eval` dos
tensores acontece na thread que gerou, antes de enfileirar — a regra de stream por thread
que `prefixes.py:113` já paga hoje.

**Invalidação e corrupção.** A chave dobra modelo, stamp, política por camada (formatos
do `QuantizedKVCache`, strides), span efetivo e `layout_version`. Formato antigo nunca
casa — sem camada de compatibilidade, os arquivos velhos morrem no LRU. Arquivo truncado
ou ilegível é `UnreadableCache` → tratado como ausente → o `reach` para ali e o resto é
prefill; a linha do índice morre junto. Divergência de ids no payload é o mesmo miss.

**Concorrência.** Write-once + rename atômico: dois daemons no mesmo diretório escrevem
os mesmos bytes sob a mesma chave, e o segundo rename é um replace de conteúdo idêntico.
Leitura durante escrita nunca vê arquivo parcial.

**Latência de um hit frio, estimada** (leitura NVMe ~5 GB/s + ~0,2 ms de
open/mmap/checagem por arquivo; conferir na fase 2):

| retomada de 15k | arquivos | bytes lidos | leitura | reprefill equivalente |
|---|---|---|---|---|
| Qwen3.8-27B | 58 + 1 âncora | 0,98 GB + 154 MB | ~240 ms | 15k tokens de prefill de um 27B — ordem de 10 s |
| Lightning | 58 + 1 âncora | 92 MB + 49 MB | ~50 ms | ordem de 3-8 s (MoE A3B prefilla rápido; ainda 2 ordens acima) |
| Laguna XS | 58 | 683 MB com `keep` (2,5 GB sem) | ~170 ms (~550 ms sem) | ordem de 3-8 s |

A margem é de duas ordens de grandeza nas três famílias; mesmo errando a banda do SSD por
2x e o prefill por 2x, o hit frio ganha. O que a fase 2 mede de verdade: TTFT com hit
frio por arm, e o overhead por arquivo — se os ~240 arquivos de uma conversa longa de
Lightning (1,6 MB cada) pesarem no open/mmap, empacotar spans em segmentos é uma
otimização declarada e medida, não v1.

**O que sobrevive a quê:**

| evento | linhas (`Rows`) | âncoras (`Snapshot`) |
|---|---|---|
| unload/reload do modelo | ficam (store é do daemon) | ficam |
| restart limpo (drain) | ficam (write-behind + drain) | ficam (drain) |
| crash | ficam até o último write-behind | **perdem-se** → híbrido reprefilla no turno seguinte |

---

## 7. Fases

F1-F3 são um branch só: entre elas o daemon não é releasável (F1 desliga o mecanismo
velho antes de F2 repor o disco).

**Fase 0 — `Layout` e o round-trip por camada.** `layout` em `LayerCache` e nas classes;
`stored()`/`restore` na base (offset-only), nas cinco que faltam e no ring em ordem
absoluta; base `Composite`; protocolo `Layouts` + laguna declarando as sliding.
Verificação: round-trip por camada (`stored` → fatia por span → concat → `restore` →
estado igual), sem modelo, no espírito de `tests/test_prompt_cache.py`; ring
promovido → `stored` absoluto → `restore` → re-promovido, igual. Mutação por caminho.

**Fase 1 — `core/prefix.py` em memória, nos dois caminhos, com âncoras.** `Chain`,
`PrefixStore` com `resume`/`commit`, resolução de layout contra o span, split do último
bloco de prefill na fronteira, captura e supersessão das duas âncoras. Entram juntos:
`language.begin_batch`/`BatchPrefill`/`batching.step`/`TextBatch.finish` e
`generate.stream_ids`. Somem no mesmo passo: `_boundary`, `standing`, `promoted`, os três
`insert`, a condição tripla de `kept_prefix`, o gate de compilação contra a trie,
`prefix_cache()`, `StandingLayer`, `_grown`. Verificação: `resume` + prefill do resto
**vs** prefill inteiro, logits completos, fp32 `< 1e-5` com `MLX_ENABLE_TF32=0` antes de
importar MLX, por família (uma KV pura, laguna, uma DeltaNet/Mamba, deepseek_v4,
longcat); divergência no meio; compartilhamento concorrente (duas sequências no mesmo
tick adotando os mesmos spans); segundo turno de híbrido com `max_tokens=1` cobrindo até
a âncora de prefill.

**Fase 2 — `Vault` e residência.** `Held`/`Filed`, LRU, tetos, write-behind, drain;
`server/prefixes.py` vira `FileVault` (`DiskSpill` e `Spill` somem) e a tabela do
`store.py` encolhe. Verificação: reuso sobrevive a restart limpo; arquivo truncado é
miss; teto respeitado; latências do §6 medidas.

**Fase 3 — Deleção.** `core/prompt_cache.py` inteiro, papéis, `Budget`/`Ledger`/weakrefs,
`storable()`, o `Spill[Any]` das options. `is_trimmable` fica: quem usa é a especulação.

**Fase 4 — Especulação.** Remove a recusa de `generate.py:471` e o `None` de
`language.py:505-510`; truncamento do draft na fronteira (single e `_speculative_tick`);
por último a chain própria do `Persistent`, com o assert como guarda até lá.
Verificação: retomada não muda aceitação — mesma seed, mesmo prompt, com e sem hit →
`acceptance` igual e saída igual.

**Fase 5 — Medição e validação de ponta a ponta.** O §11, mais um `omnia-bench paired`
com gate térmico provando decode neutro (streams idênticos) — o commit anda no laço de
decode e não pode custar tok/s.

---

## 8. Correção

Invariantes:

1. Um `restore` deixa **todas** as camadas no mesmo offset (a âncora). Offsets
   divergentes é a classe de bug que decodifica fluente em cima de estado que nunca
   existiu.
2. Snapshot só é capturado com a camada parada na fronteira. Fora dela, não existe.
3. Os ids do span vão no payload e são conferidos em toda leitura. Colisão de hash não é
   risco aceitável aqui; divergência é miss e o arquivo é esquecido.
4. A chave dobra modelo, stamp, política por camada, span efetivo e versão de layout.
   Formato antigo nunca casa.
5. Zero-fill só existe sob `keep`, e `keep` só é declarável onde a máscara de janela
   prova que as linhas puladas têm peso zero em toda leitura futura (attend mascarado e
   `promote` da janela).
6. Um snapshot superado por descendente na mesma cadeia é largado, nunca filado.
7. `commit` nunca dentro de uma rodada especulativa (`rows > len(tokens)` recusa).
8. A âncora de prefill não é evictada enquanto a de decode existir (ordem de pressão).

Testes, com mutação obrigatória por caminho numérico novo:

- round-trip por camada (fase 0);
- paridade `resume`+prefill vs prefill inteiro, logits completos, por família (fase 1);
- segundo turno híbrido via âncora de prefill, `max_tokens=1` (fase 1);
- retomada não muda aceitação especulativa (fase 4);
- mutações, cada uma tendo que ficar vermelha: quebrar o encadeamento da chave; trocar o
  `stride` do pool; remover a re-rotação do ring no `restore`; capturar snapshot fora da
  fronteira; mover `commit` para dentro da rodada; zero-fill além do `keep` declarado;
  supersessão apagando a âncora de prefill em vez da de decode.

---

## 9. Riscos assumidos

- **VRAM duplicada enquanto o pedido roda.** O span copiado convive com o buffer contíguo
  vivo. Limitado pelo teto, que despeja. A alternativa é atenção paginada, que trocaria o
  caminho de atenção e os kernels por um gather por passo — fora de escopo, e
  provavelmente pior no decode.
- **Cauda parcial perdida**: `<= span - 1` tokens por turno, mais o que o adaptador
  re-renderiza (thinking descartado, tool calls re-serializados) — esse segundo termo é
  do cliente, não do cache, e a sonda do §11 o separa.
- **Sliding como `Rows` custa 3x o útil no store** (laguna: 120 KB/token contra 40).
  Poda por camada exigiria mutar spans compartilhados com refcount por cadeia — fora do
  v1. `keep` corta a leitura; o armazenamento paga o teto e uma conversa laguna de 62k
  não cabe no disco default (§5, §6).
- **Crash perde as âncoras** → híbrido frio reprefilla uma vez. Restart limpo não perde
  nada (drain).
- **Divergência no meio do histórico num híbrido é reprefill**: sem pontos intermediários
  de retomada, editar um turno antigo volta ao começo para as camadas `Snapshot` (as
  `Rows` retomam até a âncora disponível, que nesse caso não existe abaixo da
  divergência). O tráfego do Claude Code é monotônico por sessão; se outro tráfego
  pesar aqui, âncoras extras são a extensão natural — medidas antes.
- **`room == 0` na rodada especulativa** deixa uma fronteira sem âncora de decode (~1/span
  das rodadas): custo de até um span de cobertura, só para cliente que ecoa ids.

---

## 10. O que é deletado (o produto)

- `core/prompt_cache.py` inteiro: trie por token, `Reuse`, `Role`, `_EVICTION_ORDER`,
  `Ledger`, `victim()`, `Budget` com weakrefs, `Spill`.
- `core/cache.py`: `is_storable`; o `signature` de rotação do ring (o de formato do
  `QuantizedKVCache` fica — vira política da chave). `cache_file.py`: `storable()`; `key`
  sobre ids inteiros vira a chave encadeada.
- `generate.py`: `_boundary`, `standing`, `promoted`, os três `insert`, o `kept_prefix`
  triplo, o gate de compilação contra a trie, a recusa da especulação.
- `language.py`: `prefix_cache()`, `StandingLayer`, `_grown`, o insert do
  `TextBatch.finish` e o `kept_prefix` incondicional.
- `server/prefixes.py`: `DiskSpill`, o `recall` que varre todos os arquivos, `_common`,
  `_packed`/`_unpacked`, o `_SPILL_FLOOR`.
- `server/store.py`: `ids` e `tokens` de `PrefixCacheFile`.
- `GenerationOptions.prefix_spill: Spill[Any]` — o único `Any` desse caminho morre com
  ele; as options passam a carregar o store do daemon, sem generics.
- Duas delegações manuais de cache composto (`DSACache`, `DeepseekV4Cache`),
  substituídas por `Composite`.

E uma exclusão que deixa de existir: especulação e prefixo passam a compor.

---

## 11. Ponto de validação: Claude Code de ponta a ponta

O aceite final não é uma suíte nem um `omnia-bench`: é o Claude Code falando com o
daemon, em três modelos, todos com ganho de TTFT. O critério é do dono do repo e fica.

### Por que este cliente

Claude Code reenvia a conversa inteira a cada turno — system prompt e definições de tool
(grandes e estáveis), histórico, resultados de tool anexados no fim. É um prefixo que só
cresce, com muitos turnos curtos sobre um prompt grande, então o tempo percebido é quase
todo TTFT. E é o cliente que expõe o caso que a versão anterior deste plano errava: o
adaptador descarta thinking e re-serializa tool calls, então o turno N+1 estende o
**prompt** do turno N, não os ids que o modelo gerou — é a âncora de prefill que
trabalha, e um plano sem ela validaria só clientes que ecoam ids.

### Os três arms

| modelo | família | classes de cache | âncora | o que o arm prova |
|---|---|---|---|---|
| Qwen 3.8 27B NVFP4 (`local/Qwen3.8-27B-nvfp4`) | `qwen3_5` | 48 DeltaNet (`Snapshot`) + 16 attention (`Rows`) | 154 MB | o maior estado dos três: supersessão de âncora sem nenhum byte de snapshot no SSD durante a sessão |
| Nemotron 3.5 Lightning 30B-A3B NVFP4 | `nemotron_h` | 23 Mamba2 (`Snapshot`) + 6 attention (`Rows`) + 23 MoE (`{}`) | 49 MB | híbrido MoE com MTP: capping da rodada + camadas de layout vazio |
| Laguna XS 2.1 NVFP4 (`poolside/Laguna-XS-2.1-NVFP4-mlx`) | `laguna` | 10 full (`Rows`) + 30 sliding (`Rows(keep=512)`) | 0 | ring promovido × commit em ordem absoluta, zero-fill sob `keep`, e o maior volume de linhas por token |

Nenhum dos três é KV puro e denso — o conjunto é deliberadamente a ponta difícil. O
caminho fácil já está coberto pela paridade por família da fase 1.

Span 256 nos três, e nos três o ganho tem que aparecer **no segundo turno**. Um híbrido
não espera conversa longa: a âncora de prefill é capturada na última fronteira do prompt
do turno 1 e consumida de VRAM pelo turno 2 — inclusive no replay com `max_tokens=1`,
que é o que o split do último bloco garante.

### Protocolo

Sessão viva não é comparável consigo mesma: a saída do modelo vira o prompt do turno
seguinte, então dois arms divergem no segundo turno e passam a medir trabalho diferente.

1. **Gravar uma vez.** Capturar os corpos de `/v1/messages` de uma sessão real
   (`server/anthropic.py`) num transcript. Uma gravação, replicada idêntica em todos os
   arms.
2. **Replay determinístico.** Alimentar os corpos na ordem, `max_tokens=1`, medindo TTFT
   por requisição. O sampler deixa de importar porque os prompts vêm da gravação.
3. **Interleaved e com gate térmico** (skill `measurement`): A (cache off,
   `prefix_cache_bytes=0`) / B (cache on) / A / B, nunca todos os A e depois todos os B.
4. **Três estados de B**: VRAM quente; só-disco (`DELETE /admin/prefixes/memory`, que já
   existe em `server/state.py:145`, mantendo o disco); e frio de restart (daemon
   reiniciado limpo após drain — o estado que valida o §6).
5. **Uma sessão viva**, num dos três, para exercer o commit do lado do decode, a
   travessia de fronteira e o capping especulativo — que o replay com `max_tokens=1` não
   toca.
6. **Decode neutro**: `omnia-bench paired` contra `main`, gate térmico, streams
   idênticos — o custo do commit no laço de decode tem que ficar dentro do ruído.

### O que registrar por requisição

- TTFT;
- cobertura: `reused_tokens` (o meter já tem o campo, `generate.py:309-315`) contra o
  esperado — que **não** é "prompt menos 255": é a última fronteira do prompt do turno
  anterior, porque o sufixo novo (turno assistant re-renderizado + tool results + turno
  do usuário) é prefill legítimo;
- **qual tier respondeu** — VRAM, vault, ou miss — e bytes lidos/escritos no vault,
  separando linhas de âncora (campos novos no `Meter`, ao lado de `reused_tokens`;
  o plano esconde a residência do *chamador*, não do meter);
- posição e residência da âncora usada no hit;
- ocupação dos dois tetos ao fim de cada turno.

### Aceite

Um critério, os três modelos, sem exceção:

- **TTFT cai em todo turno depois do primeiro, nos três**, nos três estados de B —
  inclusive no só-disco, onde a leitura do vault tem que ganhar do reprefill (§6).
- **Zero bytes de âncora no SSD durante a sessão** nos dois híbridos: âncora é superada
  em memória e só é filada no drain. Centenas de MB/s de snapshot no disco = supersessão
  quebrada.
- **Nenhum arm muda a saída.** Mesmo prompt gravado, mesmo primeiro token. A igualdade
  forte (logits, fp32) é da fase 1; aqui é sanidade de ponta a ponta.

Se um híbrido não ganhar do segundo turno em diante, o defeito está numa destas três,
nesta ordem: o split do último bloco não parou na fronteira (§2.3), a supersessão evictou
a âncora errada (§8.8), ou o prefixo do cliente não é estável entre turnos — que é o que
a sonda abaixo descarta antes de qualquer medição.

### O diagnóstico que precisa existir antes de medir

Uma sonda de divergência de prefixo: para requisições consecutivas do transcript, o
prompt **renderizado** do turno N tem que ser prefixo estrito do turno N+1 até o fim do
prompt de N. Se o adaptador reordenar tools ou injetar qualquer coisa instável, a cadeia
quebra no primeiro id diferente e todas as medições acima dão zero por um motivo que não
tem nada a ver com o cache. No miss, logar o comprimento do prefixo comum contra a
fronteira esperada — é o que separa "o cache não funciona" de "o cliente nunca manda o
mesmo prefixo duas vezes", e o que mede o custo real do thinking descartado.
