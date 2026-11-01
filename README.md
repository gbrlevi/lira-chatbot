# Lira: agente de atendimento da Empório da Música (PoC)

A Empório da Música é uma loja fictícia de instrumentos musicais em Campo Grande/MS, com a equipe sobrecarregada por perguntas recorrentes sobre horários, pedidos, preços e estoque.

Esta PoC é a **Lira**, uma assistente virtual de texto em Python que decide sozinha quando consultar dados da loja e quando consultar as políticas, atende em pt-BR e encaminha para a equipe o que precisa de uma pessoa.

O princípio que guiou tudo: **regras de dinheiro, estoque e identidade ficam no código, não no prompt.** O modelo conversa e escolhe ferramentas. Quem calcula e verifica é o código.

```
Cliente: Quanto custa o Yamaha C40? dá pra parcelar em 12x?
  consultando buscar_produtos
  consultando simular_parcelamento
Lira: O Yamaha C40 Nylon Natural custa R$ 599,90, ou R$ 569,91 no PIX.
      Em 12x não é possível para esse valor, mas dá para dividir em até 6x sem juros de R$ 99,98.
```

Há conversas completas em [`src/examples/`](src/examples/).

## Como rodar

**Pré-requisitos:** [uv](https://docs.astral.sh/uv/) (ele instala o Python 3.12) e uma chave de api de ia do google. Ela pode ser obtida gratuitamente no [Google AI Studio](https://aistudio.google.com/apikey).

```bash
uv sync
cp .env.example .env      # depois preencha AI_API_KEY no .env
uv run lira-ingest        # cria a base; baixa cerca de 240 MB do modelo de embeddings na primeira vez
uv run lira               # conversa no terminal
```

Modos úteis da CLI:

```bash
uv run lira --debug              # mostra as ferramentas chamadas e o que retornaram
uv run lira --thread minha-conversa   # retoma uma conversa depois de fechar
uv run lira -m "Qual o endereço da loja?"   # uma única mensagem
```

API, o ponto de entrada que um canal como o WhatsApp chamaria:

```bash
uv run lira-api
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "Quanto custa o Takamine GD20?"}'
```

A resposta traz `thread_id`; envie-o de volta para continuar a conversa. Em `http://127.0.0.1:8000/docs` está a documentação interativa.

Também há `docker compose up --build`

### Testes e avaliações

```bash
uv run pytest                                   # não precisa de chave nem de rede
uv run python src/evals/run_agent_evals.py      # usa o modelo real, cerca de 110 requisições
uv run python src/evals/retrieval_benchmark.py  # compara estratégias de busca
```

### Configuração

Tudo fica em `.env`; veja [`.env.example`](.env.example). As mais úteis são `LLM_MODEL`, para trocar de modelo, e `REFERENCE_DATETIME`, que fixa o relógio para reproduzir um cenário.

## Arquitetura

```
              ┌──────────────┐
 mensagem ──▶ │ input_guard  │  tamanho, sem LLM
              └──────┬───────┘
                     ▼
              ┌──────────────┐  chamadas   ┌───────────────────────┐
              │    agent     │ ◀────────▶ │        tools          │
              │ (Gemini)     │             │ produtos, pedidos,    │
              └──────┬───────┘             │ políticas, parcelas,  │
                     ▼                     │ chamados              │
              ┌──────────────┐             └───────────┬───────────┘
              │ output_guard │  valores em R$          │
              └──────┬───────┘  têm origem?            ▼
                     │ (regenera 1x)         ┌───────────────────┐
                     ▼                       │ SQLite: catálogo, │
              ┌──────────────┐               │ FTS5 e vetores    │
              │ aviso de     │               └───────────────────┘
              │ expediente   │
              └──────┬───────┘
                     ▼
                  resposta
```

| Ferramenta | Para quê | O que o código garante |
|---|---|---|
| `buscar_produtos` | Preço, estoque, promoção, recomendação | Filtros rígidos; preço final já calculado; lista só o que pode ser comprado |
| `detalhar_produtos` | Especificações e comparação | Ignora descrições que contradizem o produto |
| `resumir_catalogo` | Visão geral, marcas, faixas de preço | Cobre os agregados que justificariam SQL livre |
| `simular_parcelamento` | Parcelas permitidas | Faixas mínimas da política; recusa valor que ninguém informou |
| `consultar_pedidos` | Status e rastreio | Verifica identidade, não revela o motivo da falha, limita tentativas |
| `consultar_politicas` | Trechos do manual | Busca híbrida, com a seção de origem |
| `registrar_chamado` | Encaminhar à equipe | Protocolo, sem duplicar chamados |

## Decisões técnicas

Cada decisão abaixo explica o problema, a escolha, o que descartei e o que aceitei perder. Quando há medição, cito o número; quando é julgamento meu, sem medição, digo isso.

| Tema | Escolha | Motivo principal |
|---|---|---|
| Arquitetura | Agente único com ferramentas | Os assuntos se misturam na mesma pergunta, e as ferramentas já fazem o roteamento |
| Tipo de modelo | LLM de fronteira por API, sem modelo local | Escolha de ferramentas confiável, sem exigir GPU, pagando só pelo uso |
| Modelo | `gemini-3.5-flash-lite` | O mais barato e rápido dos testados, e o trabalho difícil já está no código |
| Orquestração | LangGraph, com pouco LangChain | Memória, retentativa e ciclo de ferramentas prontos, com o fluxo à vista |
| Dados da loja | Ferramentas determinísticas, sem SQL livre | O código aplica as regras de preço e status em toda consulta |
| Armazenamento | SQLite, FTS5 e NumPy | Dados pequenos, um processo só e nenhum servidor para instalar |
| Políticas | Busca híbrida por seção | Clientes misturam termos exatos e paráfrases |
| Embeddings | `fastembed` local | Sem rede nem cota; o Gemini mede melhor, mas os evals não falharam por retrieval |
| Guarda-corpos | Em código, sem LLM extra | Determinísticos, testáveis e sem custo por mensagem |
| Avaliação | Checagens determinísticas e leitura das respostas | Reprodutível, mas só a leitura encontrou os erros mais sutis |

### Agente único com ferramentas, sem multiagente

A alternativa natural seria um roteador enviando cada mensagem a um especialista em catálogo, pedidos ou políticas. Ela funciona mal aqui porque os assuntos vêm juntos: "quanto custa e parcela em 12x?" ou "posso devolver o pedido 8?" cruzam domínios na mesma frase e exigiriam re-roteamento. Os especialistas também teriam a mesma persona, as mesmas regras e o mesmo histórico, então a separação duplicaria prompt e somaria chamadas de LLM sem isolar nada de verdade.

Com um único agente, escolher a ferramenta é o próprio roteamento: a pergunta acima vira duas chamadas em um turno, com 2 a 3 chamadas ao modelo por turno. **O custo aceito** é um prompt único que concentra todas as regras. Se o escopo crescer a ponto de as ferramentas competirem entre si, a separação fica simples, porque cada ferramenta já é independente.

### LLM de fronteira por API, sem modelo local

O modelo precisa fazer três coisas: entender português informal, escolher a ferramenta certa com argumentos válidos e redigir a resposta sem alterar o que a ferramenta devolveu. A segunda é a mais crítica, porque um erro ali leva o cliente a receber uma informação errada com aparência de certa. Escolhi um modelo por API pelos motivos abaixo:

- **Menor risco técnico para começar:** modelos de fronteira chegam com chamada de ferramentas pronta e bem testada. Um modelo aberto também poderia servir, mas somaria decisões sem relação com o problema da loja: qual modelo, qual quantização, qual servidor e como ele formata as ferramentas.
- **Qualquer pessoa consegue rodar:** bastam uma chave gratuita e um computador comum. Um modelo local bom o bastante pediria vários GB de download e, para responder rápido, uma GPU; o resultado da PoC passaria a depender da máquina de quem a executa.
- **Custo proporcional ao uso:** um caso de avaliação custa cerca de US$ 0,004 no plano pago, então 1.000 conversas curtas ficariam em torno de US$ 4. Um servidor com GPU cobra pelo mês inteiro, inclusive de madrugada e com a loja fechada. Só compensaria com um volume muito acima do que uma loja como esta gera.
- **Foco no que a PoC precisa provar:** a pergunta é se o agente resolve o atendimento, e operar um servidor de modelos é outro projeto.

**O custo aceito** tem duas partes. A primeira é depender de um provedor, e isso apareceu nos testes: houve timeouts e erros 503 em outro modelo da mesma família. Por isso há timeout de 30 s, até 3 tentativas apenas em falhas transitórias e uma mensagem amigável quando o provedor segue fora. A segunda é que as mensagens saem da máquina: no free tier, o Google informa que [pode usar o conteúdo para melhorar seus produtos](https://ai.google.dev/gemini-api/docs/pricing). Isso só é aceitável porque os dados são fictícios; em produção, seria obrigatório usar o plano pago, em que esse uso não ocorre.

Eu reavaliaria um modelo local diante de volume alto ou de uma exigência contratual ou de LGPD que impeça enviar dados a terceiros. A suíte de avaliação torna essa comparação direta. Trocar de modelo Gemini é uma variável; trocar de provedor exige um pacote novo e poucas linhas em `llm.py` e `resilience.py`.

### Modelo: `gemini-3.5-flash-lite`

A arquitetura reduz o que se exige do modelo: conta, estoque, identidade e datas estão no código, e ele só precisa escolher ferramentas e redigir. Sendo assim, o critério passa a ser **o modelo mais barato e rápido que dê conta do trabalho**, e não o mais capaz. Velocidade e preço pesam porque cada turno faz de 2 a 3 chamadas, e cada caso de avaliação processa cerca de 11,9 mil tokens de entrada, entre prompt, ferramentas, histórico e retornos. Na média, a entrada responde por 93% do custo.

| Modelo | Preço por 1M de tokens (entrada/saída) | Custo estimado por caso* | Mediana por caso* | Limite diário gratuito | Resultado |
|---|---|---|---|---|---|
| **`gemini-3.5-flash-lite`** | US$ 0,30 / 2,50 | US$ 0,003 | 2,4 s | 500 requisições | Ver [Qualidade](#qualidade) |
| `gemini-3.5-flash` (hoje [encaminhado ao 3.6](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5)) | US$ 0,75 / 3,75; US$ 1,50 / 7,50 a partir de 2027 | US$ 0,011; US$ 0,021 a partir de 2027 | 7,4 s | 20 requisições | 11 de 11 casos antes de bater no limite |
| `gemini-3.8-flash` | US$ 0,75 / 3,75; US$ 1,50 / 7,50 a partir de 2027 | Sem dado | Sem dado | 20 requisições | Timeouts de 90 s e erros 503; nenhum dado de qualidade |

**Nos mesmos cenários, o flash-lite saiu cerca de 3 vezes mais barato e 3 vezes mais rápido que o `gemini-3.5-flash`.** Parte da diferença vem da saída: 144 tokens por caso contra 855, provavelmente por causa do raciocínio interno, que é cobrado como saída. Em um atendimento por mensagens, essas diferenças contam a cada turno, e o raciocínio extra compensa pouco quando as regras difíceis já estão no código.

A cota gratuita maior veio como vantagem adicional: a suíte de testes faz cerca de 110 chamadas por rodada, então dá para iterar sem pagar.

**Há duas ressalvas.** A primeira é que a comparação é parcial: os 11 casos do Flash são quase todos de catálogo, e o limite gratuito impediu rodar a suíte inteira nos modelos maiores. Isso sustenta preço e velocidade, mas **não prova que o flash-lite tem a melhor qualidade**; só mostra que a qualidade dele atende a esta PoC, conforme os resultados em [Qualidade](#qualidade). A segunda é que o preço do Flash deve dobrar em 2027, levando a diferença de custo para cerca de 6 vezes, se nada mudar.

O modelo ignora `temperature` porque usa amostragem fixa, então esse parâmetro não é configurado. Por isso cada caso de avaliação roda duas vezes: o mesmo pedido pode gerar respostas diferentes.

### LangGraph com LangChain

Um loop de chamadas de ferramentas cabe em poucas linhas com o SDK do Gemini, e para um agente sem estado eu usaria exatamente isso. Este precisa de mais:

- voltar ao modelo depois das ferramentas, inclusive com várias chamadas no mesmo turno;
- passar por um verificador que pode pedir uma nova resposta uma única vez;
- retomar a conversa depois de fechada;
- repetir só a chamada ao modelo, só em erro transitório, sem executar as ferramentas de novo.

Com o SDK puro, eu escreveria e testaria tudo isso à mão. **O LangGraph já traz** `SqliteSaver`, `RetryPolicy` por nó, `ToolNode` e `recursion_limit`, e os guarda-corpos entram como nós comuns. O fluxo inteiro cabe nas 141 linhas de [`graph.py`](src/app/agent/graph.py), lidas de cima a baixo. Usei `StateGraph` em vez de `create_agent`, do LangChain 1.x, que seria mais curto, porque ele esconderia justamente os desvios para o verificador, que são o centro do projeto. Frameworks multiagente, como o CrewAI, resolvem um problema que não existe aqui.

**Do LangChain uso apenas o necessário:** `init_chat_model`, o decorador `@tool` e os tipos de mensagem. Eles geram o esquema das ferramentas a partir das funções, deixam a troca de provedor barata e ligam o rastreamento do LangSmith por variável de ambiente. Não uso chains, retrievers nem vector stores: a busca híbrida é código próprio, simples de medir e depurar.

**O custo aceito** é uma dependência grande, que muda rápido e às vezes atrapalha. Dois exemplos reais daqui: a resposta do modelo chega em blocos, exigindo `.text` em vez de `.content`, e o cliente embrulha os erros do provedor, então `is_transient` precisa percorrer a cadeia de exceções. O `uv.lock` fixa as versões para que isso não mude sozinho.

### Dados estruturados: ferramentas determinísticas, sem SQL livre

Um `top-k` semântico não responde "violões até R$ 1000", que tem exatamente 12 respostas certas: ele devolveria um número fixo de itens parecidos. Por isso RAG não serve para catálogo, e o caminho são filtros de verdade.

Considerei uma ferramenta de SQL "livre" como último recurso e **testei antes de decidir**: em 19 perguntas de estresse, nenhuma falha seria resolvida por SQL. Todas vinham de dados mal normalizados (`steel` contra "aço") ou de semântica. Os únicos ganhos seriam agregados raros, cobertos por `resumir_catalogo` ou compostos a partir de listas, pois os 61 itens disponíveis cabem em cerca de 900 tokens.

Do outro lado, "só em último caso" não é garantível, porque o modelo escolhe pela descrição da ferramenta, e um `SELECT` sem filtro recomendaria um produto descontinuado ou uma promoção expirada. Com ferramentas parametrizadas, essas regras valem em toda consulta. **O custo aceito** é que uma pergunta que nenhuma ferramenta previu não tem resposta; ela vira um chamado, e isso registra a necessidade real. Em [PoC e produção](#poc-e-produção) está o desenho seguro, caso os chamados a confirmem.

### Armazenamento: SQLite + FTS5 + NumPy, deliberadamente para a PoC

Os dados são 65 produtos, 25 trechos do manual e algumas tabelas de pedidos, lidos por um único processo. O SQLite cobre isso com um arquivo só e sem instalar nada, pois já vem com o Python:

- **Filtros de verdade:** "até R$ 1000, com cordas de nylon, disponíveis" é uma consulta SQL com junções. Com pandas ou listas em memória, eu reimplementaria parte disso.
- **Busca textual incluída:** o FTS5 oferece BM25 no mesmo arquivo, sem outro serviço.
- **Vetores sem índice aproximado:** com tão poucos vetores, a busca com cosseno exato em NumPy leva 6 ms na mediana, incluindo gerar o embedding da pergunta. Um índice aproximado traria complexidade para resolver um problema de escala que não existe.
- **Recriável:** `lira-ingest` reconstrói `emporio.db` a partir de `data/`. Conversas e chamados ficam em `runtime.db`, então ingerir de novo não apaga atendimentos.

Descartei o **Postgres com pgvector** nesta fase porque exigiria um servidor antes da primeira conversa, sem trazer ganho para esse volume. Descartei **Chroma ou FAISS** porque seriam um segundo armazenamento para manter em sincronia e não trazem os filtros relacionais nem o BM25 do FTS5.

**Em produção eu usaria Postgres com pgvector.** O SQLite admite um único escritor por vez, não atende bem a várias instâncias da aplicação e seu FTS5 não tem stemming em português, enquanto o Postgres resolve os três pontos e tem backup e operação maduros. A troca fica contida na camada de armazenamento: ferramentas e agente não precisam mudar de contrato.

### Retrieval: híbrido por seção, com medição

**Extração:** o PDF vira Markdown com `pdfplumber`, preservando tabelas. Testei o `pdftotext -layout`, e ele separava cada dia do seu horário, o que quebraria a pergunta mais comum da loja. Testes de fidelidade garantem que fatos críticos sobrevivam à extração.

**Chunks por seção:** o manual já é dividido em regras numeradas, com uma ideia por subseção, e cada uma virou um trecho, somando 25. Janelas de tamanho fixo poderiam separar uma regra da sua condição, e a seção de origem ajuda o agente a se situar. A busca devolve 4 trechos: incluir um a mais custa poucos tokens, enquanto omitir a regra certa leva a uma resposta errada.

**Combinação com RRF:** BM25 e cosseno têm escalas sem relação entre si, e o Reciprocal Rank Fusion evita calibrá-las, pois usa só a posição no ranking. Mantive o `k=60`, o valor usual, sem ajuste fino, para não superajustar a 40 perguntas.

Benchmark com 40 perguntas escritas à mão, 4 trechos retornados:

| Estratégia | Recall@4 | Recall@1 | MRR | Latência mediana |
|---|---|---|---|---|
| BM25 (FTS5) | 0,950 | 0,775 | 0,850 | 1 ms |
| Vetorial, fastembed | 0,725 | 0,575 | 0,642 | 6 ms |
| **Híbrido com RRF, fastembed (padrão)** | 0,950 | 0,675 | 0,785 | 8 ms |
| Vetorial, Gemini Embedding | 0,950 | 0,825 | 0,881 | 477 ms |
| Híbrido com RRF, Gemini Embedding | 0,975 | 0,775 | 0,869 | 511 ms |

**Só por esse benchmark, o BM25 sozinho venceria o híbrido com fastembed:** mesmo recall, melhor ordenação e nenhum modelo para baixar. Mantive o híbrido por dois motivos. O BM25 erra paráfrases abertas, como "Onde vocês ficam?" e "Comprei pela internet e não gostei, e agora?", que é como clientes de fato escrevem, embora o híbrido erre outras duas perguntas, como "O produto chegou amassado na entrega". E os vetores já são necessários para ordenar recomendações de produtos, como "violão para viagem". Com uma diferença tão pequena em 40 perguntas, a amostra não decide a questão, e um conjunto maior poderia mudar esta escolha.

### Embeddings: `fastembed` local, mesmo com o Gemini medindo melhor

O Gemini Embedding é claramente superior ao modelo local. Mantive o `fastembed` porque a Lira lê os 4 trechos, então a ordenação pesa menos que o recall; porque a diferença é de 1 pergunta em 40 numa amostra pequena; e porque os evals de políticas não falharam por retrieval. Em troca, a busca dispensa rede e cota, leva 8 ms em vez de 511 ms e os testes rodam sem chave.

Entre as opções locais, o `fastembed` usa ONNX e dispensa o PyTorch, que seria uma dependência pesada para só gerar embeddings em CPU. O modelo `paraphrase-multilingual-MiniLM-L12-v2` tem suporte a português, mas **trunca em 128 tokens**, o que afeta os chunks maiores; o BM25 cobre o texto inteiro. Trocar custa uma variável: `EMBEDDINGS_PROVIDER=gemini`, seguida de `lira-ingest`.

### Semântica só ordena, nunca decide o que existe

Previ um limiar de similaridade para "saxofone" retornar vazio, e **a medição derrubou a ideia**: "trompete" (0,54) pontua acima de "baixo para banda" (0,37), e "qual a capital da França" (0,37) acima de "aceita pix?" (0,19). Nenhum corte separa os casos. Dizer que a loja tem um produto que não existe é pior que não recomendar nada, então a existência vem de critérios rígidos: categoria, modelo com busca aproximada no nome ou palavra literal no texto. A semântica só ordena o que já passou por eles.

### Prompts modulares em XML, gerados com avaliação

O prompt da Lira fica em cinco arquivos XML em [`src/prompts/`](src/prompts/), separados por responsabilidade: persona, escopo, regras de atendimento, uso de ferramentas e formato de saída. A divisão torna cada mudança fácil de revisar em um diff, e as tags separam com clareza as regras dos dados. Regras críticas aparecem no início e no fim, por causa da tendência do Gemini de perder instruções no meio de contextos longos. O bloco com data e hora vem por último, para manter o restante estável.

A persona se chama Lira por remeter à música, fala como "um amigo que entende de música" e **sempre se declara assistente virtual**, porque o cliente tem o direito de saber com quem fala. O prompt fica em português porque essa é a língua da conversa, o que evita o modelo traduzir regras e termos da loja.

A lição mais útil veio dos testes: **regras negativas funcionaram mal neste modelo, e exemplos concretos funcionaram bem.** "Não prometa prioridade" não impediu promessas; um exemplo de resposta correta reduziu o problema, embora ele ainda reapareça, como mostra [Qualidade](#qualidade). Quando nem isso bastou, a regra foi para o código.

### Guarda-corpos em código, sem LLM extra

Um segundo LLM auditando cada mensagem somaria custo e latência a todo turno e também poderia errar. Preferi cobrir com código o que dá para verificar deterministicamente, porque uma regra em código não depende de o modelo obedecer:

- **Valores:** o manual avisa que informação incorreta "pode configurar propaganda enganosa", então dinheiro recebeu a proteção mais forte. Todo R$ da resposta precisa constar em um retorno de ferramenta ou na fala do cliente. Senão, a resposta é descartada e regenerada uma vez; persistindo, sai uma resposta segura e fixa. A política é **fail-closed**, porque não informar um preço custa menos ao cliente e à loja do que informar um preço errado.
- **Injeção:** conteúdo do cliente e das ferramentas é tratado como dado. Mais importante que isso, as ferramentas não expõem dados de terceiros, então uma instrução maliciosa não tem o que extrair, mesmo que convença o modelo.
- **Entrada e expediente:** o limite de 2.000 caracteres barra mensagens abusivas sem gastar uma chamada, e o aviso de loja fechada é aplicado em código, porque o modelo o omitia mesmo com a regra no prompt.

**O custo aceito** é que o código só cobre o que é verificável com precisão: uma condição inventada em texto passa, e essa é a principal limitação declarada abaixo. Para produção, a proposta está em [PoC e produção](#poc-e-produção).

### Identidade: telefone mais pedido ou e-mail, verificados na ferramenta

Dados de pedido são pessoais, então a verificação fica dentro de `consultar_pedidos`, onde nenhuma instrução ao modelo consegue pulá-la. O telefone é o fator principal porque é único, enquanto dois clientes dividem o mesmo e-mail. O segundo fator existe porque um telefone sozinho é fácil de obter. Falhas devolvem sempre o mesmo erro genérico, para não revelar qual dado estava certo, e três tentativas erradas bloqueiam a consulta naquela conversa.

### Memória: checkpointer em SQLite, com janela

O cliente pode fechar e retomar a conversa, como acontece em um canal de mensagens, então memória em processo não bastaria. O `SqliteSaver` persiste cada passo sem servidor adicional.

O modelo recebe só os 12 turnos mais recentes, cortados sempre no início de um turno, o que nunca separa uma chamada de ferramenta do seu resultado. O número 12 é julgamento meu, não medição: cobre atendimentos longos e impede o custo de crescer sem limite. Não há sumarização, porque atendimentos são curtos, e resumir custaria LLM e arriscaria perder telefone e número do pedido.

### Interface: CLI e API finas, sem streaming de tokens

Ambas são camadas sobre o mesmo `ChatService`: a CLI serve para demonstrar e depurar, com `--debug`, e a API é o ponto que um canal real chamaria. A CLI mostra o progresso das ferramentas em tempo real, mas o texto só aparece depois de aprovado, porque transmitir tokens mostraria ao cliente uma resposta que o verificador ainda pode descartar. Com respostas em poucos segundos, o ganho do streaming não compensa esse risco.

### Avaliação: checagens determinísticas, sem LLM como juiz

Cada caso de avaliação verifica ferramentas usadas, fatos obrigatórios e termos proibidos. Um LLM juiz avaliaria nuance melhor, mas gastaria cota, variaria entre rodadas e poderia errar do mesmo jeito que o agente. Checagens por texto são reprodutíveis e, quando falham, mostram exatamente o motivo.

**O custo aceito** é que elas às vezes reprovam uma resposta aceitável ou aprovam uma resposta ruim. Por isso as respostas também são lidas manualmente, e foi assim que surgiram os erros mais sutis, listados em [Qualidade](#qualidade).

### Ambiente e bibliotecas de apoio

| Escolha | Por quê | Alternativa descartada |
|---|---|---|
| Python 3.12 | Versão madura, com pacotes prontos para as dependências nativas (`onnxruntime`, `numpy`) | Versões mais novas, com maior risco de faltar pacote pronto |
| `uv` | Instala o próprio Python e fixa as versões em um lockfile, então a instalação é reproduzível em poucos comandos | `pip` com `requirements.txt`, sem lockfile completo |
| `pydantic-settings` | Configuração tipada e validada ao iniciar, em um lugar só | `os.environ` espalhado pelo código |
| `rapidfuzz` | Encontra "GD 20" e erros de digitação em nomes de modelos sem outro serviço | Exigir o nome exato do produto |
| FastAPI | Validação de entrada e documentação em `/docs` com muito pouco código | Flask, que exigiria validação e documentação à parte |
| Rich | Uma CLI legível, que mostra o que o agente está fazendo | `print` simples, pior para demonstrar |
| Docker opcional | Quem usa `uv` não precisa dele, e quem prefere contêiner tem a opção | Docker obrigatório, um atrito sem ganho para uma PoC com SQLite |

## Tratamento dos dados

Inspecionar os dados revelou problemas que levariam o agente a errar:

| Achado | Tratamento |
|---|---|
| Sem coluna de marca; separar o nome gerava "Bateria", "Teclado" e "Music" | Marca reconhecida por uma lista, com erro explícito se surgir uma nova |
| Especificações em inglês e mistas: `steel`, `Black` e `Preto` | Normalizadas na ingestão |
| Em `strings`, guitarras têm a contagem e violões o material | Separado em `string_type` e `string_count` |
| **Descrições que contradizem o produto:** o Korg Synth 1 Pro se diz "Roland, 88 teclas", mas tem 61 | Seis descrições que citam outra marca ficam fora da busca e não chegam ao agente; nome e especificações valem |
| Produto ativo sem estoque, descontinuado com estoque e "em breve" com vendas | Status vence estoque; só ativo com estoque é vendável |
| Só 4 de 25 promoções ativas, com produtos repetidos | Só ativas contam, com validação de no máximo uma por produto |
| Pedidos 3 e 20 com total diferente da soma dos itens | Vale o total registrado |
| Dois clientes com o mesmo e-mail | Telefone é o fator de identidade |
| Categorias sem produtos, e "mais de 300 instrumentos" no manual contra 65 no CSV | O agente diz "não encontrei no sistema", nunca "não vendemos" |
| A palavra "cordas": acessório, violão de 7 cordas e cordas orquestrais | Distinção explícita no prompt e nos testes |

## Suposições

Pontos ambíguos do manual, resolvidos pelo critério de que o específico vence o geral:

- **Contatos:** (67) 3341-4444 é o telefone geral e (67) 3321-4500 o WhatsApp, pois a seção 7 é específica; e-mail sem acento, como na capa.
- **Parcelas:** as faixas da seção 3.1 valem sobre a tabela da seção 3; "exceto abaixo de R$ 50,00" é lido como parcela mínima de R$ 50,00 até 3x.
- **Frete de exatamente R$ 500,00:** não coberto; o agente orienta confirmar com a equipe, sem prometer frete grátis.
- **Arrependimento:** vale para compras online, e os pedidos não registram o canal, então o agente pergunta.
- **Prazos de entrega:** não há data real de recebimento; a previsão serve de aproximação, sempre com ressalva.
- **Lojas parceiras:** o manual pede para sugerir, mas não cita nenhuma; o agente não inventa.
- **Preço de produto descontinuado ou "em breve":** não é exibido, para não soar como oferta.
- **Data atual:** usa o relógio real em `America/Campo_Grande`. Como os pedidos vão até março de 2026, entregas aparecem muito atrasadas; `REFERENCE_DATETIME` fixa o relógio em testes e exemplos.

## Qualidade

| Camada | O que cobre | Resultado |
|---|---|---|
| Testes (`pytest`) | Dados, regras, ferramentas, grafo, guardas, memória e API, sem rede | 319 passando |
| Evals com o modelo real | Ferramenta correta, fatos obrigatórios e proibidos, em 51 cenários, duas rodadas | 100 de 102 execuções aprovadas |
| Benchmark de retrieval | Tabela acima | Recall@4 de 0,950 |

| Categoria | Aprovadas |
|---|---|
| Catálogo | 34 de 34 |
| Parcelamento | 8 de 8 |
| Políticas | 22 de 22 |
| Pedidos | 16 de 16 |
| Escopo | 4 de 4 |
| Segurança | 10 de 10 |
| Atendimento | 6 de 8 |

A rodada não teve erros de provedor, fez 222 chamadas ao modelo e ficou com mediana de 2,6 s por caso; a execução mais lenta levou 31 s. Os resultados completos, com todas as respostas, estão em [`src/evals/results/`](src/evals/results/).

**A falha desta rodada é real e continua aberta:** em `handoff_complaint` (2 de 2 execuções), a Lira abre o chamado e informa protocolo e prazo certos, mas promete cuidar do caso "com prioridade", o que a loja não oferece. A regra e o exemplo no prompt reduziram o problema sem eliminá-lo. A próxima tentativa seria tratar isso no retorno de `registrar_chamado`, como foi feito com o PIX.

**O resultado varia entre rodadas, porque o modelo não é determinístico.** A rodada anterior, com o código quase igual, deu 98 de 100 e falhou em outros dois casos, que passaram nesta:

- `orders_withdrawal_applied_to_a_real_order`: com telefone e pedido na mensagem, a Lira citou só a política, não consultou o pedido e ofereceu registrar a devolução, sem notar que o prazo provavelmente já tinha passado. É a falha mais séria observada, porque o cliente sai com uma expectativa errada.
- `policy_withdrawal_right`: a Lira perguntou se a compra foi online antes de citar os 7 dias. Mantive a checagem estrita, mas considero esse comportamento aceitável, porque o direito depende do canal.

Por isso, leia esses números como a ordem de grandeza de uma amostra pequena, sem tratá-los como uma taxa de acerto estável.

**Os números sozinhos enganam, e isso foi a principal lição de engenharia aqui.** Rodadas com 100% de aprovação escondiam erros reais, que só apareceram lendo as respostas. Cada um virou uma correção e um teste:

| Erro observado com o modelo real | Causa | Correção |
|---|---|---|
| "10x de R$ 249,00" para um produto de R$ 2.199,00 | Modelo chamou as parcelas junto com a busca e **inventou o valor**; o verificador aceitou, pois vinha de uma ferramenta | Ferramenta recusa valor que ninguém informou |
| "Temos trompetes e trombones" | Resultado vazio incentivou preencher com conhecimento geral | Busca vazia devolve as categorias que têm produtos |
| "Previsão 25/02", sendo 28/02 | Modelo errou ao converter a data | Datas saem prontas da ferramenta |
| "Anotei seu telefone", sem ter anotado | Chamado existente ignorava o contato novo | Contato passa a ser gravado |
| "Não fazemos reservas", regra que não existe | Nenhum caso cobria o assunto. Depois, o caso novo aprovava a resposta errada numa rodada de 100 de 100, porque a checagem não reconhecia a frase | Ferramenta orienta consultar a política e traz uma frase-modelo para o que o manual não cobre; checagem endurecida; 7 perguntas manuais sobre assuntos fora do manual responderam certo |
| "R$ 417,24 no PIX" para um produto em promoção | Depois de ver um preço com PIX na conversa, o modelo copiou o formato e fez a conta. O verificador barrou o valor, mas às vezes o cliente ficava sem preço, e os evals de um turno não mostravam isso | Retorno da ferramenta declara que o PIX não se aplica; caso novo com dois turnos nos evals |
| Aviso de loja fechada omitido | Regra no prompt não era obedecida | Aviso aplicado em código |

## Limitações conhecidas

- **Condições inventadas sobre políticas:** o verificador em código só cobre valores em R$. Uma condição falsa em texto, como "após passar pela nossa análise", depende do prompt. Em uma rodada anterior, a Lira também disse que um pedido confirmado "está em preparação para envio", algo que o sistema não informa, e na publicada escreveu uma vez "entregue em 17/02" para o que é só a previsão. É o maior risco residual.
- **Oferece o que não consegue fazer:** não há ferramenta de compra nem de cálculo de frete, mas às vezes ela oferece "fechar o pedido", "garantir o seu" ou o valor do frete para um CEP. Se o cliente aceita a compra, é orientado a ir à loja ou usar os contatos dela.
- **Ausência de regra lida como negativa:** quando o manual não trata de um assunto, como reservas, agora ela diz que não tem a informação. Isso depende de o modelo seguir o aviso da ferramenta, sem garantia em código.
- **Resposta fixa quando o verificador barra duas vezes:** se o modelo insiste em um valor sem origem, o cliente recebe a resposta segura e fica sem o preço, mas não recebe um errado.
- **Erros de ortografia ocasionais** do modelo, como "registreia".
- **Identidade por dados conhecidos:** telefone mais número do pedido é um controle fraco, adequado só a uma PoC.
- **Limite de tentativas por conversa:** abrir outra conversa zera a contagem.
- **Feriados:** não há calendário, então o aviso de expediente os ignora.
- **Sem autenticação na API**, e o histórico guarda dados pessoais sem política de retenção.
- **Conjuntos de teste pequenos e escritos à mão:** bons para regressão, insuficientes para afirmar taxas de acerto em produção.
- **Cota:** o free tier limita 15 requisições por minuto e 500 por dia, por projeto, renovando à meia-noite do horário do Pacífico. Acima disso, o cliente recebe a mensagem de indisponibilidade, e rodadas repetidas de avaliação esgotam o limite diário.
- **Dados no free tier:** o Google informa que pode usar o conteúdo enviado nesse plano para melhorar seus produtos. Aqui os dados são fictícios; com clientes reais, seria preciso usar o plano pago.

## PoC e produção

O que eu mudaria com mais tempo ou em produção:

- **Banco:** Postgres com pgvector no lugar do SQLite, inclusive para checkpoints e chamados.
- **Canal:** no WhatsApp, o número do remetente preencheria o telefone automaticamente, ainda editável, já que a pessoa pode consultar uma compra feita para um parente. O segundo fator continuaria obrigatório.
- **Gatekeeper dedicado:** um classificador binário antes do agente, com um prompt especializado nas regras de bloqueio e saída 0 ou 1. Eu avaliaria primeiro o [Jev](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/), da TypeSafe AI, um modelo de decisão não generativo que devolve escolhas tipadas com probabilidade, cujo caso ideal é justamente classificar entre poucas classes. A fabricante declara até 200x mais velocidade e 400x menos custo que LLMs. Relatos independentes são mistos, inclusive um teste com menor acurácia que um LLM pequeno. Por isso, a adoção dependeria de um benchmark com um conjunto adversarial próprio, e os controles em código continuariam ativos.
- **Verificação de condições:** um auditor de saída para cobrir o maior risco residual, além dos valores em R$.
- **SQL de catálogo, se os chamados mostrarem necessidade:** banco separado e sem dados pessoais, `set_authorizer`, limite de linhas e tempo, e views que já apliquem as regras de status e promoção.
- **Chamados:** integração com CRM no lugar da tabela local.
- **Observabilidade:** o rastreamento do LangSmith já funciona por variável de ambiente; faltam métricas de conversa e alertas.
- **Embeddings:** reavaliar o Gemini Embedding com um conjunto maior, e testar reranking.

## Uso de assistentes de código

Usei o **Claude Code** como assistente de programação durante o projeto, sempre decidindo a arquitetura e tomando todas as decisões. O fluxo foi:

1. **Planejamento:** antes de qualquer código, analisei os requisitos e os dados, e tomei logo de inicio as decisões mais relevantes, como arquitetura, modelo, armazenamento, regras de negócio...
2. **Decisões humanas que mudaram o rumo:** pensei em uma ferramenta de SQL como último recurso e pedi uma análise crítica de sua necessidade, o que levou ao teste que a descartou. Também troquei o modelo recomendado pelo `gemini-3.5-flash-lite`, por já ter utilizado diversos modelos em produção e já ter a noção de que essa é uma das opções mais baratas e rápidas, condicionando a escolha a um teste com dados reais da loja.
3. **Prompts:** gerados com uma skill propria, um pipeline de engenharia de prompts com pontuação em 12 dimensões. Revisei cada versão do prompt principal e fiz alguns ajustes finos.
4. **Implementação:** o assistente escreveu a maior parte do código e testes, com minha revisão e validação acontecendo a cada etapa.
5. **Verificação:** o mais valioso foi ler as respostas reais do agente, além das métricas. Os erros da tabela em [Qualidade](#qualidade) só foram encontrados assim.

Toda a responsabilidade pelas decisões e pelo resultado é minha.

## Estrutura

```
data/              CSVs da loja e manual de políticas
src/app/           código: ingest, retrieval, domain, tools, agent, interfaces
src/prompts/       módulos XML do prompt da Lira
src/tests/         testes sem rede
src/evals/         casos, avaliação com o modelo real e benchmarks
src/examples/      conversas reais capturadas do agente
```
