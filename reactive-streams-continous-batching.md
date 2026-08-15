# Um decode: contínuo, compilado, para todos os modelos

> **Estado da implementação (2026-08-14, segunda leva — commit `0371814`).** As 16
> famílias que restavam no caminho single migraram: **45/45 famílias com
> `continuous_batching = True`**. O que a leva precisou do core, e só isso: `softcap`
> na porta `attend` (softmax manual na ordem exata do gemma2, ancorado por teste de
> referência próprio; `QuantizedKVCache` recusa), o protocolo `RaggedBatchable`
> (o cache da família responde pelo próprio adaptador ragged — batching.py nunca
> aprende nome de família: DSACache, LatentKVCache, MLACache, NgramCache,
> BatchedDeepseekV4Cache moram nas famílias), `BatchedLayerCache` (camadas sem estado
> dos híbridos) e `BatchedSharedKVReader` (publish por linha, gemma3n/gemma4).
> Validação da leva: paridade batched-vs-solo + isolamento por família (46 testes),
> caminho single **bit-idêntico ao HEAD** em 14/14 famílias comparáveis
> (max_abs = 0.0; bailing via suíte de forward com checkpoint; glm_moe_dsa era
> inconstruível no HEAD — bug pré-existente corrigido), mutações dos caminhos novos,
> paired bench Qwen3.8-27B-nvfp4 **neutro com streams idênticos em 128 ids**.
> Deliberadamente não ampliado: kernels Metal single-row (sink do gpt_oss, MambaStep
> fundido, MoE fundidos) seguem B=1 por guard; B>1 decodifica eager — otimização
> gated em bench. `stream_ids` agora serve apenas prompts Iterator e imagens.
>
> **Estado da implementação (2026-08-14, primeira leva).** As fases 0–5 implementadas no working
> tree, com os desvios e pendências abaixo. Validação: paridade + mutação por caminho
> novo (testes tiny, sem checkpoint), suíte completa no nível da baseline do HEAD
> (toda falha restante foi verificada pré-existente), pyright/ruff/lint-imports limpos,
> e dois paired benches reais com gate térmico — laguna-xs NVFP4 e Qwen3.6-35B oQ4e —
> ambos **neutros com streams idênticos em 128 ids**.
>
> - **Fase 0** — `core/decode.py` (DecodePlan/compiled_decode/Buckets/load_slots);
>   qwen3_5, nemotron_h, laguna e mamba2 são clientes; mamba2 ganhou decode compilado
>   de trunk inteiro (novo caminho, com paridade + mutação próprias).
> - **Fase 1** — `server/flow.py` (Outlet/Member/Clock); `_decode`/`_decode_batch`
>   colapsados em um caminho com fases Prefilling (um bloco por tick — o freeze do
>   joiner acabou), Decoding e Streaming (transitória).
> - **Fase 2** — `step()` retorna 0..k ids por sequência; reasoning budget é estado da
>   `BatchSequence` (o closer devido é alimentado e emitido como em `stream_ids`);
>   B∈{2,4,8} com slots mortos mascarados (laguna).
> - **Fase 3** — 30 famílias tier A + jamba + lfm2(dense/moe) com CB; a porta `attend`
>   ganhou `ragged_mask` (sliding por linha) e `sinks`; adaptadores
>   `BatchedDeltaCache`/`BatchedConvCache` para híbridos; bucket compilado **genérico**
>   para famílias KV-puras (B∈{1,2,4,8}, padding, regrow, residência por modelo).
>   Bloqueadas com razão documentada (xfail estrito onde há teste): qwen3_next e
>   deepseek_v2 (atenção fora da porta + B=1 nos kernels), gemma2 (idem), bailing_hybrid
>   (MLA compõe latent+k_pe no attend), falcon_h1 (cache composto por camada), e o
>   restante do tier B (gpt_oss/deepseek_v4/glm4_moe-dsa/gemma3n/gemma4/longcat) —
>   seguem no caminho single de hoje, nada regrediu.
> - **Fase 4** — `stream()` despacha batch-de-1 síncrono quando `can_batch`; o trie
>   recebe sempre a forma rewindável (fatia do buffer fixo de volta em `KVCache`).
>   `stream_ids` permanece para: prompts Iterator (arriving), imagens, famílias sem
>   flag. A deleção total do §8 continua gated pela regra de transição.
> - **Fase 5** — um request com drafter não exclui mais o modelo do CB: a rodada
>   especulativa roda por linha dentro do tick (0..k absorve os aceitos), reusando
>   `speculative._round`; linhas com drafter nunca entram nos buckets compilados (o
>   cache delas precisa rewindar). O grafo (B,k) compartilhado fica gated em bench.
> - **Correção de bug latente** encontrada pela unificação: no caminho batched, a
>   máscara da gramática era computada um token atrás do `accept` — corrigido
>   (`_shared_step` avança o matcher antes de mascarar o próximo draw).

Plano para a codebase ter **uma** maneira de decodar: `decode(ids, states, capacity)`,
sempre sob o clock de continuous batching, sempre como grafo compilado por bucket.
Sem caminho eager de decode, sem fallback ragged, sem flag por modelo, sem "modo
single". T=1 é um batch de 1 — um bucket como qualquer outro, não um código.

O objetivo primário é **simplificação**: hoje existem N maneiras de fazer a mesma
coisa, e cada uma carrega seu ciclo de vida, seus bugs e sua manutenção. O plano
lista explicitamente o que é deletado (§7), porque a deleção é o produto.

## 1. As N maneiras de hoje (o que este plano colapsa)

Caminhos de decode coexistindo:

1. `generate.stream_ids` — loop eager single-sequence (todo modelo);
2. `batching.step` genérico — `model(ids, batch(caches))` com `BatchedKVCache`
   atendendo linha a linha em Python (qwen3);
3. `single_decode`/`single_greedy`/`prepare_single_greedy` — compilado T=1 (laguna);
4. `batch_decode`/`batch_greedy` — compilado B∈{2,4} (laguna);
5. `compile_decode` com regrow+recompile — compilado T=1 crescente (qwen3_5,
   nemotron_h, e uma variação em laguna);
6. steps por bloco compilados (mamba2, bailing_hybrid, qwen3_5).

E dois ciclos de vida no server (`_decode`, `_decode_batch`), cinco protocolos de
decoder em `batching.py`, uma flag manual `continuous_batching`, e um `can_batch`
com três exclusões.

O item 5 é a prova de viabilidade do plano inteiro: `qwen3_5/model.py:109-197` já
entrega a *experiência* de cache crescente sobre grafo compilado — promove para
buffer fixo dimensionado por `fit()`, e o closure recompila sozinho quando o bucket
estoura (`regrow`) ou quando delegators mudaram (`epochs`/`stale()`). O plano
generaliza esse padrão, escrito hoje três vezes com variações, para uma
implementação em core que toda família usa.

## 2. A forma final

### 2.1 Engine: `core/decode.py` (novo) — o maquinário, uma vez

O que qwen3_5/nemotron_h/laguna duplicam vira infraestrutura genérica:

```python
def compiled_decode(
    model: DecodeStep,                 # o que a família fornece (§2.2)
    slots: Sequence[Sequence[FixedState]],
    capacity: int,
) -> Callable[[mx.array], mx.array]:
    """Um grafo por bucket (B, capacity, head_mode). Cuida de:
    - promoção de estado crescente p/ forma fixa (uma vez, na entrada do slot)
    - regrow + recompile quando base + steps >= capacity  (padrão qwen3_5)
    - staleness por epoch quando estratégias re-resolvem   (padrão qwen3_5)
    - máscara de validade `columns <= position` por linha  (padrão laguna/qwen3_5)
    - cache de grafos por residência: buckets sobrevivem entre requests
    """
```

Buckets de B: `{1, 2, 4, 8, ...}`; um batch de 3 **padda para 4** com uma linha
morta mascarada no commit. Isso é barato de propósito: decode é limitado pela
leitura dos pesos, e uma linha a mais lê os mesmos pesos — o custo marginal é ~zero,
que é o argumento físico de "sempre batchado" ser aceitável até para B=1.

Dimensões do bucket: `(B, capacity, head_mode)`, onde `head_mode` é logits ou
argmax fundido (o kernel `lm_head_argmax` do laguna) — escolhido no build pelo mix
dos membros (todos greedy sem filtro → argmax; senão logits). É uma dimensão de
despacho dentro do único caminho, não um caminho.

### 2.2 O contrato por família: um step traceável

Cada família fornece exatamente uma coisa nova — o corpo que o trace percorre:

```python
class DecodeStep(Protocol):
    def make_cache(self) -> list[LayerCache]: ...          # já existe
    def decode_step(
        self, ids: mx.array, slots: Sequence[FixedState], mask: mx.array
    ) -> mx.array: ...                                     # denso, traceável
```

Para a maioria das famílias, `decode_step` é o `activations` de hoje com a máscara
recebida em vez de construída — RMSNorm, rope, sdpa, MLP/MoE traceiam sem cerimônia
(laguna e qwen3_5 provam MoE, rope custom e delegators dentro do grafo). As regras
de traceabilidade que o laguna documenta (`model.py:366-372` — pesos fora da
captura, só estado vivo nos containers) viram doc do contrato, não redescoberta por
família.

Sem `default.py` eager para decode: **o piso de correção passa a ser o próprio
prefill** — a paridade stepwise (decode compilado vs logits do prefill, full logits)
já é a regra do projeto, e o forward de prefill/`activations` continua existindo
porque é o mesmo código que o trace percorre. Não é uma segunda maneira de decodar;
decode nunca o executa eagerly.

### 2.3 Estado: toda classe de cache tem forma fixa e sabe entrar/sair de slot

`engine/core/cache.py` e módulos de família:

- já existem: `FixedKVCache`, `RingKVCache`, `FixedDeltaCache`;
- ganham o mesmo tratamento: `ConvCache` (lfm2), `FalconH1LayerCache`,
  `DeepseekV4Cache`, `DSACache` (glm4_moe), `NgramCache` (longcat),
  `SharedKVReader` (gemma3n/gemma4), MLA (deepseek_v2);
- o protocolo é `promote` (crescente → fixo na capacity), `regrow` (bucket maior),
  `snapshot/restore` (entrar/sair de um slot compilado — o que
  `laguna._make_batch_slots`/`_load_batch_slots` fazem hoje na mão, movido para o
  estado);
- máscara e posição pertencem ao estado/kernel, nunca ao código do modelo: o braço
  `mx.array` do `laguna._sliding_mask` morre; `RingKVCache` mascara a si mesmo.

Prefill escreve **direto no buffer fixo** quando a capacity já é conhecida
(`fit(prompt + max_tokens)`), eliminando a cópia de promoção no caminho comum. O
trie de prefixo (`PromptCache`) guarda a forma rewindável — fatiar o buffer fixo em
`rows` é o snapshot que ele armazena.

### 2.4 `batching.py`: o despacho encolhe

Os cinco protocolos (`SingleDecoder`, `SingleGreedyDecoder`, `BatchDecoder`,
`BatchGreedyDecoder`, `PreparedSingleGreedyDecoder`) somem. `step()` vira:

```python
def step(model, sequences) -> list[list[int]]:      # 0..k tokens por sequência
    bucket = resolve_bucket(sequences)              # (B↑, capacity↑, head_mode)
    outputs = bucket.decode(stacked_ids)            # sempre compilado
    # sampling/penalty/constraint fora do grafo, por linha, como hoje
```

O contrato passa a **0..k tokens por sequência por tick**, o que absorve:

- reasoning budget: o closer devido (`owed` de `stream_ids`) vira estado da
  `BatchSequence`, processado no commit;
- especulação: uma **dimensão do bucket** `(B, k)`, não um caminho — o verify é um
  grafo `[B, k+1]` compilado como qualquer outro. Sobre slots fixos, o problema
  clássico da especulação batched (aceitação variável por linha) degenera: as k+1
  linhas são escritas para todo mundo, `position[i] = base[i] + aceitos_i` é um
  vetor, e as linhas sujas além da position **não são atendidas** pela máscara de
  validade — o rewind ragged vira atualização de um vetor. O bucket especulativo
  constrói quando todas as linhas são greedy-verificáveis (mesma regra de dimensão
  que `head_mode`); batch misto roda no bucket normal. Estado recorrente não tem
  máscara que o salve (acumulação sequencial): exige checkpoint no início do bloco
  e re-avanço pelos aceitos — KV puro entra primeiro, híbridos depois, e o desenho
  começa lendo como o MTP do qwen3_5 lida com o rewind do delta hoje.

`can_batch` e a flag `continuous_batching` deixam de existir — não há o que
perguntar quando só existe um caminho. `TextLanguageModel.stream()` vira um wrapper
síncrono: batch de 1 sobre `step()` até terminar. O corpo de `stream_ids` morre.

### 2.5 Server: `flow.py` + o clock (inalterado em relação à versão anterior)

- `Outlet`: entrega de segmentos; emitir nunca bloqueia o clock, fechar é
  idempotente, o async-for do consumidor termina no close (anyio memory stream,
  buffer infinito — backpressure invertida de propósito: cliente lento é derrubado,
  nunca freia o tick).
- `Intake`: fila com `take_compatible` (joiner não fura a ordem).
- `Clock[S]`: o único loop — cancelados saem, tick na model thread
  (`run_in_executor` na thread dedicada, **não** `anyio.to_thread`: o pool não
  garante a mesma thread OS e o contexto Metal/buckets vive numa específica),
  emissões roteadas, admissão até `room()`.
- Fases por membro: `Prefilling` (avança **um bloco por tick**, intercalado com o
  decode dos ativos — corrige o congelamento atual em que o prefill de um joiner
  para todo mundo) e `Decoding`.
- `_decode`, `_decode_batch`, `_account` avulso, `_queue`/`_pending` e o trio
  chunks/sentinela/state somem do `server/engine.py`.

Divisão: engine possui o tick (síncrono, sem event loop, deps mínimas — anyio nunca
entra no engine); server possui o clock (membership, admissão, outlets, métricas,
residency). Teste da fronteira: *existe numa geração única num script?* → engine.

## 3. Modelos: o que cada um precisa para ter um step traceável

45 famílias. A pergunta por família deixa de ser "suporta batching?" e vira
"o step tracea e o estado tem forma fixa?".

### Já provaram o padrão (4)

`qwen3_5`, `nemotron_h`, `laguna`, `mamba2` — têm decode compilado hoje. Alteração:
**deletar** suas cópias do maquinário (regrow/epochs/slots/buckets) e virar clientes
de `core/decode.py`. São a migração-prova: o critério de aceitação da fase 0 é
paired bench neutro nelas antes/depois.

### Tier A — forward liso, KV puro (~26)

`afmoe, apertus, bailing_moe, bitnet, cohere, ernie4_5, ernie4_5_moe, exaone4,
gemma, gemma3, glm4, gpt2, granite, hunyuan_dense, hy3, llama, mimo_v2,
muse_glimmer (trunk), olmo2, olmoe, phi3, qwen2, qwen2_moe, qwen3, seed_oss,
smollm3, step3p7`

Alteração: `decode_step` = `activations` de hoje com máscara injetada (edição
pequena e mecânica por família), + paridade parametrizada + mutation. O trace de
RMSNorm/rope/sdpa/MLP/MoE já está provado. Sliding window (gemma3) espera o
`RingKVCache` automascarante.

### Tier B — atenção/cache próprios (~9)

| família | o que precisa |
| --- | --- |
| `gemma2`, `llama4` | máscara sliding sai do modelo; step traceável |
| `gpt_oss` | sinks/sliding: o kernel próprio vira o attend do estado fixo |
| `deepseek_v2` | MLA: forma fixa + attend do cache comprimido |
| `deepseek_v4` | `DeepseekV4Cache`: forma fixa + regrow do cache indexado |
| `glm4_moe` | `DSACache` idem |
| `gemma3n`, `gemma4` | `SharedKVReader`: forma fixa preservando o compartilhamento |
| `longcat_flash_ngram` | `NgramCache` + MLA: dois estados fixos |

Trabalho por família, independente, dias cada.

### Tier C — recorrentes/híbridos restantes (~6)

`qwen3_next, jamba, bailing_hybrid, falcon_h1, lfm2` (+ o que mamba2/qwen3_5 ainda
não cobrirem). `FixedDeltaCache` existe; falta `ConvCache` fixo e
`FalconH1LayerCache` fixo. Os híbridos compõem camada a camada — o maquinário de
core não distingue KV de Delta num mesmo slot (qwen3_5 já mistura os dois num grafo).

## 4. Laguna: antes e depois

Antes: 497 linhas, das quais ~250 são maquinário — `_make_batch_slots`,
`_load_batch_slots`, `_single_bucket`, quatro dicts de buckets, cinco métodos de
decode, a flag, e o `_sliding_mask` com braço batched.

Depois:

```python
class Laguna(nn.Module):
    def make_cache(self) -> list[KVCache | RingKVCache]: ...      # igual

    def decode_step(self, ids, slots, mask) -> mx.array:
        # o corpo do forward de _compile_batch_forward, e só ele:
        x = self.model.embed_tokens(ids)
        for block, slot in zip(self.model.layers, slots, strict=True):
            x = block(x, mask, slot)
        return self.head(self.model.norm(x))
```

Somem do arquivo: a flag (não existe mais no projeto), os slots (estado sabe
entrar/sair — §2.3), os buckets e os cinco métodos (core/decode.py — §2.1), o braço
batched da máscara (§2.3). O kernel `lm_head_argmax` continua — vira o `head_mode`
do bucket, declarado onde os outros kernels do laguna já são. O arquivo volta a ser
a arquitetura, e o maquinário que só ele tinha vira o que **todas** as famílias
usam.

## 5. Fases

0. **`core/decode.py`**: extrair o maquinário de qwen3_5/nemotron_h/laguna/mamba2 e
   migrá-los como clientes, deletando as cópias. Aceitação: paridade + mutation +
   paired bench neutro nos quatro, T=1 e B=2/4 no laguna.
1. **`server/flow.py` + clock**: Outlet/Intake/Clock, fases com chunked prefill,
   `_decode`/`_decode_batch` colapsados. (Durante a transição, membros de famílias
   ainda não migradas rodam como fase `Streaming` embrulhando o `stream()` atual —
   um transitório que a fase 4 deleta.) Aceitação: testes de cancelamento
   concorrente/join/desconexão; latência de joiner medida antes/depois.
2. **Contrato 0..k + reasoning budget no commit**; buckets com padding de B.
3. **Migração por tier**: A em levas (mecânico + paridade), C (formas fixas de
   Conv/FalconH1), B (uma por vez). Cada família aceita com: paridade stepwise
   (tolerância de `noise.batching` medido), mutation no seu `decode_step`, paired
   bench não-pior que o caminho que substitui. Uma regressão de perf numa família é
   bug do maquinário a corrigir — não motivo para reter o caminho antigo.
4. **Deleção** (§7) + `stream()` como batch-de-1 síncrono.
5. **Especulação como dimensão de bucket `(B, k)`** — KV puro primeiro
   (rewind-por-máscara), híbridos depois (checkpoint de estado). Antes de desenhar:
   ler como o MTP do qwen3_5 resolve o rewind do delta em B=1. O gate "spec ativo
   enquanto B ≤ N" só é criado se o paired bench mostrar que a composição
   spec×batch deixa de pagar em algum B — N vem de medição, não de intuição.

## 6. Riscos assumidos (e por que são aceitáveis)

- **Retrace em mudança de membership/bucket**: mitigado por buckets persistentes
  por residência e padding (B=3 usa o grafo de 4 que já existe). O retrace de
  regrow já é pago hoje pelo qwen3_5 e amortiza.
- **Leitura do buffer cheio por step**: custo de banda proporcional à capacity do
  bucket, não ao usado. Já é o custo do decode compilado atual; `fit()` escalona.
  Medir por família no paired bench — é o número que decide se o maquinário precisa
  de buckets de capacity mais finos.
- **Prompts curtos / gerações curtas**: pagam trace na primeira request de um
  bucket novo e nada depois (grafos persistem). O prefill direto no buffer fixo
  (§2.3) remove a cópia de promoção do caminho comum.
- **Famílias com step difícil de tracear**: o custo não desaparece — é deslocado
  para onde é pago uma vez (contrato documentado + maquinário pronto) em vez de
  redescoberto por família.
- **Justiça entre modelos residentes**: hoje o gate serializa *gerações inteiras*
  (um modelo espera a geração completa do outro). Com o clock, a unidade de
  intercalação cai para o tick (~ms), e todo tick é limitado por construção —
  decode e verify custam ~uma leitura de pesos, prefill é limitado pelo bloco.
  Round-robin de ticks entre clocks resolve a justiça; nenhum gate adicional é
  pré-construído. Exceção a medir: verify espec em **MoE** lê até ~(k+1)× os bytes
  de expert de um step (menos, com sobreposição de roteamento) — 2–3× um tick
  normal. Se o paired bench mostrar pressão sob multi-modelo, o k é o knob, e a
  política é uma linha no clock (server), sem tocar engine.

## 7. Invariante: nenhuma feature é removida

O plano deleta **implementações duplicadas**, nunca capacidades. Toda feature
visível de hoje tem endereço no estado final — este é o checklist de aceitação da
fase 4 (nada dela conclui com um item órfão):

| feature | onde vive depois |
| --- | --- |
| samplers, penalty, grammar/constraint | fora do grafo, por linha, no commit do `step()` (como hoje) |
| stop tokens, `max_tokens`, clamp de `context_limit` | estado da `BatchSequence` / `prepare` |
| reasoning budget (closers devidos) | estado da `BatchSequence`, processado no commit (§2.4) |
| prompt arriving (iterator) + prefill sobreposto ao produtor | prefill bloco a bloco (§2.3) alimentado pelo iterator; a fase `Prefilling` consome blocos conforme chegam |
| prefix cache (`PromptCache` trie), `Budget`, `Spill` | inalterados; o trie guarda a forma rewindável — fatia do buffer fixo em `rows` (§2.3) |
| especulação MTP (qwen3_5, nemotron_h) e DFlash (muse_glimmer) | dimensão de bucket `(B, k)` (§2.4); até a fase 5, seguem no caminho atual — a exclusão de `can_batch` só cai quando o substituto estiver verde |
| visão/multimodal (gemma3n, gemma4, qwen3_5, step3p7…) | o encoder de visão roda no prefill (embeddings), o decode é texto — mesmo tick; até a família migrar, fase `Streaming` transitória |
| KV cache quantizado (`QuantizedKVCache`, via `quantizing.py`) | mais uma classe de estado com forma fixa/`stacked` (§2.3); até lá, `Streaming` |
| meters/métricas, cancelamento, SSE incremental | `on_join`/`on_leave` + `Outlet` no clock (§2.5) |
| API do bench (`prepare_batch_sequence`/`step` em `bench/arms/omnia.py`) | mantida — o bench já usa o caminho que vira o único |

A regra de transição vale para tudo: **nenhum caminho atual é desligado antes de o
substituto passar paridade + mutation + paired bench**. A fase `Streaming` existe
exatamente para isso — o que ainda não migrou continua funcionando como hoje, e a
deleção é o último passo de cada item, nunca o primeiro.

## 8. O que é deletado (o produto)

- `generate.stream_ids` (o loop eager inteiro, ~200 linhas com promoção/boundary/
  owed inline);
- `BatchedKVCache` e o attend ragged linha a linha;
- os cinco protocolos de decoder em `batching.py` e seu despacho de ~90 linhas;
- as três cópias do maquinário regrow/epochs/slots (qwen3_5, nemotron_h, laguna)
  e os quatro dicts de bucket do laguna;
- a flag `continuous_batching` e o `can_batch` com suas exclusões;
- `_decode`, `_decode_batch`, `_account` avulso, `_queue`/`_pending`,
  chunks/sentinela/state no server;
- a fase transitória `Streaming` (fim da fase 4).

Regras transversais (CLAUDE.md, valendo dobrado): paridade antes de bench;
tolerância vem de `noise.batching` medido; todo caminho numérico novo (cada
`decode_step`, cada forma fixa, o 0..k, o chunked prefill, o padding de B) ganha
mutation test; nenhuma família troca de caminho sem paired bench.
