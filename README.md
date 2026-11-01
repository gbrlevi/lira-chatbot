# Lira: customer service agent for Empório da Música (PoC)

English | [Português](README.pt-BR.md)

Empório da Música is a fictional musical instrument store in Campo Grande, Brazil, whose staff is overloaded with recurring questions about opening hours, orders, prices and stock.

This PoC is **Lira**, a text-based virtual assistant written in Python. She decides on her own when to look up store data and when to look up the policies, serves customers in Brazilian Portuguese, and hands over to the staff whatever needs a person.

The principle behind everything: **rules about money, stock and identity live in the code, not in the prompt.** The model converses and chooses tools. The code does the calculating and the checking.

```
Cliente: Quanto custa o Yamaha C40? dá pra parcelar em 12x?
  consultando buscar_produtos
  consultando simular_parcelamento
Lira: O Yamaha C40 Nylon Natural custa R$ 599,90, ou R$ 569,91 no PIX.
      Em 12x não é possível para esse valor, mas dá para dividir em até 6x sem juros de R$ 99,98.
```

Here the customer asks for the price of a Yamaha C40 and whether it can be paid in 12 installments. Lira gives the price, the price with PIX (Brazil's instant payment system), and explains that 12 installments are not available for this amount, but that up to 6 interest-free installments of R$ 99,98 are.

The agent's language is Portuguese, so messages and replies quoted in this document stay in Portuguese, with an English translation where the meaning matters. Brazilian real amounts keep the Brazilian format, as the agent writes them. Full conversations are in [`src/examples/`](src/examples/).

## How to run

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (it installs Python 3.12) and a Google AI API key. You can get one for free from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
uv sync
cp .env.example .env      # then fill in AI_API_KEY in .env
uv run lira-ingest        # builds the database; downloads a model of about 240 MB on the first run
uv run lira               # chat in the terminal
```

Useful CLI modes:

```bash
uv run lira --debug              # shows the tools called and what they returned
uv run lira --thread minha-conversa   # resumes a conversation after closing it
uv run lira -m "Qual o endereço da loja?"   # a single message
```

The API, the entry point a channel such as WhatsApp would call:

```bash
uv run lira-api
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"message": "Quanto custa o Takamine GD20?"}'
```

The response includes `thread_id`; send it back to continue the conversation. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

There is also `docker compose up --build`.

### Tests and evaluations

```bash
uv run pytest                                   # needs neither a key nor a network
uv run python src/evals/run_agent_evals.py      # uses the real model, about 110 requests
uv run python src/evals/retrieval_benchmark.py  # compares search strategies
```

### Configuration

Everything lives in `.env`; see [`.env.example`](.env.example). The most useful settings are `LLM_MODEL`, to switch models, and `REFERENCE_DATETIME`, which pins the clock to reproduce a scenario.

## Architecture

```
              ┌──────────────┐
  message ──▶ │ input_guard  │  size, no LLM
              └──────┬───────┘
                     ▼
              ┌──────────────┐  calls      ┌───────────────────────┐
              │    agent     │ ◀────────▶ │        tools          │
              │ (Gemini)     │             │ products, orders,     │
              └──────┬───────┘             │ policies, tickets,    │
                     ▼                     │ installments          │
              ┌──────────────┐             └───────────┬───────────┘
              │ output_guard │  R$ amounts             │
              └──────┬───────┘  have a source?         ▼
                     │ (retries once)        ┌───────────────────┐
                     ▼                       │ SQLite: catalog,  │
              ┌──────────────┐               │ FTS5 and vectors  │
              │ closed-store │               └───────────────────┘
              │ notice       │
              └──────┬───────┘
                     ▼
                  response
```

| Tool | Purpose | What the code guarantees |
|---|---|---|
| `buscar_produtos` | Price, stock, promotions, recommendations | Hard filters; final price already calculated; lists only what can be bought |
| `detalhar_produtos` | Specifications and comparison | Ignores descriptions that contradict the product |
| `resumir_catalogo` | Overview, brands, price ranges | Covers the aggregates that would justify free-form SQL |
| `simular_parcelamento` | Allowed installments | The policy's minimum tiers; refuses an amount nobody provided |
| `consultar_pedidos` | Status and tracking | Verifies identity, does not reveal why a check failed, limits attempts |
| `consultar_politicas` | Manual excerpts | Hybrid search, with the source section |
| `registrar_chamado` | Hand over to the staff | Protocol number, no duplicate tickets |

## Technical decisions

Each decision below explains the problem, the choice, what I ruled out and what I accepted losing. Where there is a measurement, I give the number; where it is my own judgment, unmeasured, I say so.

| Topic | Choice | Main reason |
|---|---|---|
| Architecture | Single agent with tools | Topics mix within the same question, and the tools already do the routing |
| Model type | Frontier LLM through an API, no local model | Reliable tool selection, no GPU required, paying only for usage |
| Model | `gemini-3.5-flash-lite` | The cheapest and fastest of those tested, and the hard work is already in the code |
| Orchestration | LangGraph, with little LangChain | Memory, retries and the tool cycle ready-made, with the flow in plain sight |
| Store data | Deterministic tools, no free-form SQL | The code applies the price and status rules on every query |
| Storage | SQLite, FTS5 and NumPy | Small data, a single process and no server to install |
| Policies | Hybrid search by section | Customers mix exact terms and paraphrases |
| Embeddings | Local `fastembed` | No network or quota; Gemini measures better, but the evals did not fail because of retrieval |
| Guardrails | In code, no extra LLM | Deterministic, testable and with no per-message cost |
| Evaluation | Deterministic checks and reading the answers | Reproducible, but only reading found the subtlest errors |

### Single agent with tools, no multi-agent

The natural alternative would be a router sending each message to a catalog, orders or policies specialist. It works poorly here because topics come together: "how much is it and can I pay in 12 installments?" or "can I return order 8?" cross domains within one sentence and would require re-routing. The specialists would also share the same persona, the same rules and the same history, so the split would duplicate prompts and add LLM calls without truly isolating anything.

With a single agent, choosing the tool is the routing itself: the question above becomes two tool calls in one turn, with 2 to 3 model calls per turn. **The accepted cost** is a single prompt holding all the rules. If the scope grows to the point where tools compete with one another, splitting is simple, because each tool is already independent.

### Frontier LLM through an API, no local model

The model has to do three things: understand informal Portuguese, choose the right tool with valid arguments, and write the answer without altering what the tool returned. The second is the most critical, because a mistake there gives the customer wrong information that looks right. I chose a model through an API for these reasons:

- **Lower technical risk to start with:** frontier models come with tool calling ready and well tested. An open model could also work, but it would add decisions unrelated to the store's problem: which model, which quantization, which server and how it formats tools.
- **Anyone can run it:** a free key and an ordinary computer are enough. A good enough local model would need a download of several GB and, to respond quickly, a GPU; the PoC's result would come to depend on the machine of whoever runs it.
- **Cost proportional to usage:** an evaluation case costs about US$ 0.004 on the paid plan, so 1,000 short conversations would come to around US$ 4. A GPU server charges for the whole month, including overnight and while the store is closed. It would only pay off at a volume far above what a store like this generates.
- **Focus on what the PoC needs to prove:** the question is whether the agent handles customer service, and running a model server is another project.

**The accepted cost** has two parts. The first is depending on a provider, and this showed up in testing: there were timeouts and 503 errors on another model of the same family. That is why there is a 30 s timeout, up to 3 attempts on transient failures only, and a friendly message when the provider stays down. The second is that messages leave the machine: on the free tier, Google states that it [may use the content to improve its products](https://ai.google.dev/gemini-api/docs/pricing). This is only acceptable because the data is fictional; in production, the paid plan, where this use does not occur, would be mandatory.

I would reconsider a local model given high volume, or a contractual or LGPD (Brazil's data protection law) requirement that prevents sending data to third parties. The evaluation suite makes that comparison straightforward. Switching Gemini models is one variable; switching providers takes a new package and a few lines in `llm.py` and `resilience.py`.

### Model: `gemini-3.5-flash-lite`

The architecture reduces what is asked of the model: math, stock, identity and dates are in the code, and it only has to choose tools and write. That makes the criterion **the cheapest and fastest model that can do the job**, not the most capable one. Speed and price matter because each turn makes 2 to 3 calls, and each evaluation case processes about 11.9 thousand input tokens, across prompt, tools, history and tool results. On average, input accounts for 93% of the cost.

| Model | Price per 1M tokens (input/output) | Estimated cost per case | Median per case | Free daily limit | Result |
|---|---|---|---|---|---|
| **`gemini-3.5-flash-lite`** | US$ 0.30 / 2.50 | US$ 0.003 | 2.4 s | 500 requests | See [Quality](#quality) |
| `gemini-3.5-flash` (currently [routed to 3.6](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5)) | US$ 0.75 / 3.75; US$ 1.50 / 7.50 from 2027 | US$ 0.011; US$ 0.021 from 2027 | 7.4 s | 20 requests | 11 of 11 cases before hitting the limit |
| `gemini-3.8-flash` | US$ 0.75 / 3.75; US$ 1.50 / 7.50 from 2027 | No data | No data | 20 requests | 90 s timeouts and 503 errors; no quality data |

**On the same scenarios, flash-lite came out about 3 times cheaper and 3 times faster than `gemini-3.5-flash`.** Part of the difference comes from output: 144 tokens per case versus 855, probably because of internal reasoning, which is billed as output. In a messaging service these differences count on every turn, and the extra reasoning pays off little when the hard rules are already in the code.

The larger free quota came as an added benefit: the test suite makes about 110 calls per run, so it is possible to iterate without paying.

**There are two caveats.** The first is that the comparison is partial: the 11 Flash cases are almost all catalog cases, and the free limit prevented running the whole suite on the larger models. This supports price and speed, but **does not prove flash-lite has the best quality**; it only shows its quality is sufficient for this PoC, according to the results in [Quality](#quality). The second is that the Flash price is due to double in 2027, taking the cost difference to about 6 times, if nothing changes.

The model ignores `temperature` because it uses fixed sampling, so that parameter is not set. That is why each evaluation case runs twice: the same request can produce different answers.

### LangGraph with LangChain

A tool-calling loop fits in a few lines with the Gemini SDK, and for a stateless agent that is exactly what I would use. This one needs more:

- going back to the model after the tools, including several calls in the same turn;
- passing through a verifier that can ask for a new answer exactly once;
- resuming the conversation after it was closed;
- retrying only the model call, only on a transient error, without running the tools again.

With the bare SDK, I would write and test all of this by hand. **LangGraph already provides** `SqliteSaver`, per-node `RetryPolicy`, `ToolNode` and `recursion_limit`, and the guardrails come in as ordinary nodes. The whole flow fits in the 141 lines of [`graph.py`](src/app/agent/graph.py), read from top to bottom. I used `StateGraph` rather than LangChain 1.x's `create_agent`, which would be shorter, because it would hide precisely the detours to the verifier, which are the center of the project. Multi-agent frameworks such as CrewAI solve a problem that does not exist here.

**From LangChain I use only what is needed:** `init_chat_model`, the `@tool` decorator and the message types. They generate the tool schema from the functions, keep switching providers cheap, and turn on LangSmith tracing through an environment variable. I do not use chains, retrievers or vector stores: the hybrid search is my own code, simple to measure and debug.

**The accepted cost** is a large dependency that changes fast and sometimes gets in the way. Two real examples from here: the model's response arrives in blocks, requiring `.text` instead of `.content`, and the client wraps provider errors, so `is_transient` has to walk the exception chain. `uv.lock` pins the versions so this does not change on its own.

### Structured data: deterministic tools, no free-form SQL

A semantic `top-k` cannot answer "violões até R$ 1000" (acoustic guitars up to R$ 1000), which has exactly 12 correct answers: it would return a fixed number of similar items. That is why RAG does not fit the catalog, and real filters are the way to go.

I considered a "free-form" SQL tool as a last resort and **tested before deciding**: across 19 stress questions, no failure would have been solved by SQL. They all came from poorly normalized data (`steel` versus "aço", Portuguese for steel) or from semantics. The only gains would be rare aggregates, covered by `resumir_catalogo` or composed from lists, since the 61 available items fit in about 900 tokens.

On the other side, "only as a last resort" cannot be guaranteed, because the model chooses from the tool's description, and an unfiltered `SELECT` would recommend a discontinued product or an expired promotion. With parameterized tools, these rules hold on every query. **The accepted cost** is that a question no tool anticipated has no answer; it becomes a ticket, which records the real need. [PoC and production](#poc-and-production) has the safe design, should tickets confirm that need.

### Storage: SQLite + FTS5 + NumPy, deliberately for the PoC

The data is 65 products, 25 manual excerpts and a few order tables, read by a single process. SQLite covers this with one file and nothing to install, since it ships with Python:

- **Real filters:** "up to R$ 1000, with nylon strings, available" is a SQL query with joins. With pandas or in-memory lists, I would be reimplementing part of that.
- **Text search included:** FTS5 provides BM25 in the same file, with no other service.
- **Vectors with no approximate index:** with so few vectors, exact cosine search in NumPy takes a median of 6 ms, including embedding the question. An approximate index would add complexity to solve a scale problem that does not exist.
- **Rebuildable:** `lira-ingest` rebuilds `emporio.db` from `data/`. Conversations and tickets live in `runtime.db`, so ingesting again does not erase customer interactions.

I ruled out **Postgres with pgvector** at this stage because it would require a server before the first conversation, with no gain at this volume. I ruled out **Chroma or FAISS** because they would be a second store to keep in sync and do not bring relational filters or FTS5's BM25.

**In production I would use Postgres with pgvector.** SQLite allows a single writer at a time, does not serve several application instances well, and its FTS5 has no Portuguese stemming, while Postgres solves all three and has mature backup and operations. The change stays contained in the storage layer: tools and agent do not need to change their contracts.

### Retrieval: hybrid by section, with measurement

**Extraction:** the PDF becomes Markdown with `pdfplumber`, preserving tables. I tested `pdftotext -layout`, and it separated each day from its hours, which would break the store's most common question. Fidelity tests ensure that critical facts survive extraction.

**Chunks by section:** the manual is already divided into numbered rules, with one idea per subsection, and each became an excerpt, for 25 in total. Fixed-size windows could separate a rule from its condition, and the source section helps the agent get its bearings. The search returns 4 excerpts: including one more costs few tokens, while leaving out the right rule leads to a wrong answer.

**Combination with RRF:** BM25 and cosine have unrelated scales, and Reciprocal Rank Fusion avoids calibrating them, since it uses only the position in the ranking. I kept `k=60`, the usual value, with no fine-tuning, so as not to overfit to 40 questions.

Benchmark with 40 handwritten questions, 4 excerpts returned:

| Strategy | Recall@4 | Recall@1 | MRR | Median latency |
|---|---|---|---|---|
| BM25 (FTS5) | 0.950 | 0.775 | 0.850 | 1 ms |
| Vector, fastembed | 0.725 | 0.575 | 0.642 | 6 ms |
| **Hybrid with RRF, fastembed (default)** | 0.950 | 0.675 | 0.785 | 8 ms |
| Vector, Gemini Embedding | 0.950 | 0.825 | 0.881 | 477 ms |
| Hybrid with RRF, Gemini Embedding | 0.975 | 0.775 | 0.869 | 511 ms |

**On this benchmark alone, BM25 by itself would beat hybrid with fastembed:** same recall, better ranking and no model to download. I kept hybrid for two reasons. BM25 misses open paraphrases such as "Onde vocês ficam?" (Where are you located?) and "Comprei pela internet e não gostei, e agora?" (I bought online and did not like it, now what?), which is how customers actually write, although hybrid misses two other questions, such as "O produto chegou amassado na entrega" (The product arrived dented). And vectors are already needed to rank product recommendations, such as "violão para viagem" (a guitar for traveling). With such a small difference over 40 questions, the sample does not settle the matter, and a larger set could change this choice.

### Embeddings: local `fastembed`, even with Gemini measuring better

Gemini Embedding is clearly better than the local model. I kept `fastembed` because Lira reads all 4 excerpts, so ranking matters less than recall; because the difference is 1 question in 40 on a small sample; and because the policy evals did not fail because of retrieval. In exchange, search needs no network or quota, takes 8 ms instead of 511 ms, and the tests run without a key.

Among local options, `fastembed` uses ONNX and does not need PyTorch, which would be a heavy dependency just to generate embeddings on CPU. The `paraphrase-multilingual-MiniLM-L12-v2` model supports Portuguese but **truncates at 128 tokens**, which affects the larger chunks; BM25 covers the entire text. Switching costs one variable: `EMBEDDINGS_PROVIDER=gemini`, followed by `lira-ingest`.

### Semantics only ranks, it never decides what exists

I expected a similarity threshold to make "saxofone" return nothing, and **measurement killed the idea**: "trompete" (trumpet, 0.54) scores above "baixo para banda" (bass for a band, 0.37), and "qual a capital da França" (what is the capital of France, 0.37) above "aceita pix?" (do you take PIX?, 0.19). No cutoff separates the cases. Saying the store has a product that does not exist is worse than recommending nothing, so existence comes from hard criteria: category, model with fuzzy name matching, or a literal word in the text. Semantics only ranks what has already passed them.

### Modular XML prompts, generated with evaluation

Lira's prompt lives in five XML files in [`src/prompts/`](src/prompts/), split by responsibility: persona, scope, service rules, tool use and output format. The split makes each change easy to review in a diff, and the tags clearly separate rules from data. Critical rules appear at the beginning and the end, because of Gemini's tendency to lose instructions in the middle of long contexts. The date and time block comes last, to keep the rest stable.

The persona is called Lira because the name evokes music (a lyre), speaks like "um amigo que entende de música" (a friend who knows music) and **always declares herself a virtual assistant**, because customers have the right to know who they are talking to. The prompt is in Portuguese because that is the language of the conversation, which keeps the model from translating store rules and terms.

The most useful lesson came from testing: **negative rules worked poorly on this model, and concrete examples worked well.** "Não prometa prioridade" (do not promise priority) did not stop the promises; an example of a correct answer reduced the problem, although it still reappears, as [Quality](#quality) shows. When even that was not enough, the rule moved into the code.

### Guardrails in code, no extra LLM

A second LLM auditing each message would add cost and latency to every turn and could also make mistakes. I preferred to cover with code what can be verified deterministically, because a rule in code does not depend on the model obeying it:

- **Amounts:** the manual warns that incorrect information "may constitute misleading advertising", so money got the strongest protection. Every R$ amount in the answer must appear in a tool result or in the customer's message. Otherwise, the answer is discarded and regenerated once; if the problem persists, a fixed safe answer goes out. The policy is **fail-closed**, because not giving a price costs the customer and the store less than giving a wrong one.
- **Injection:** content from the customer and from tools is treated as data. More importantly, the tools do not expose third-party data, so a malicious instruction has nothing to extract, even if it convinces the model.
- **Input and business hours:** the 2,000-character limit stops abusive messages without spending a call, and the closed-store notice is applied in code, because the model omitted it even with the rule in the prompt.

**The accepted cost** is that the code only covers what can be verified precisely: an invented condition in text gets through, and this is the main limitation stated below. For production, the proposal is in [PoC and production](#poc-and-production).

### Identity: phone plus order or email, verified in the tool

Order data is personal, so verification sits inside `consultar_pedidos`, where no instruction to the model can skip it. The phone is the main factor because it is unique, whereas two customers share the same email. The second factor exists because a phone number alone is easy to obtain. Failures always return the same generic error, so as not to reveal which piece of data was right, and three wrong attempts block lookups in that conversation.

### Memory: SQLite checkpointer, with a window

Customers can close and resume a conversation, as happens on a messaging channel, so in-process memory would not be enough. `SqliteSaver` persists each step with no additional server.

The model receives only the 12 most recent turns, always cut at the start of a turn, which never separates a tool call from its result. The number 12 is my judgment, not a measurement: it covers long interactions and stops cost from growing without limit. There is no summarization, because interactions are short, and summarizing would cost LLM calls and risk losing the phone and order number.

### Interface: thin CLI and API, no token streaming

Both are layers over the same `ChatService`: the CLI is for demonstrating and debugging, with `--debug`, and the API is the point a real channel would call. The CLI shows tool progress in real time, but the text only appears after approval, because streaming tokens would show the customer an answer the verifier may still discard. With answers in a few seconds, the gain from streaming does not make up for that risk.

### Evaluation: deterministic checks, no LLM as judge

Each evaluation case checks tools used, required facts and forbidden terms. An LLM judge would assess nuance better, but it would spend quota, vary between runs and could make the same mistakes as the agent. Text checks are reproducible and, when they fail, show exactly why.

**The accepted cost** is that they sometimes fail an acceptable answer or pass a bad one. That is why answers are also read manually, which is how the subtlest errors surfaced, as listed in [Quality](#quality).

### Environment and supporting libraries

| Choice | Why | Alternative ruled out |
|---|---|---|
| Python 3.12 | Mature version, with prebuilt packages for the native dependencies (`onnxruntime`, `numpy`) | Newer versions, with a higher risk of missing prebuilt packages |
| `uv` | Installs Python itself and pins versions in a lockfile, so installation is reproducible in a few commands | `pip` with `requirements.txt`, without a complete lockfile |
| `pydantic-settings` | Typed configuration validated at startup, in one place | `os.environ` scattered across the code |
| `rapidfuzz` | Finds "GD 20" and typos in model names with no other service | Requiring the exact product name |
| FastAPI | Input validation and documentation at `/docs` with very little code | Flask, which would need separate validation and documentation |
| Rich | A readable CLI that shows what the agent is doing | Plain `print`, worse for demonstrating |
| Optional Docker | `uv` users do not need it, and those who prefer a container have the option | Mandatory Docker, friction with no gain for a PoC using SQLite |

## Data handling

Inspecting the data revealed problems that would have led the agent into mistakes:

| Finding | Handling |
|---|---|
| No brand column; splitting the name produced "Bateria" (drums), "Teclado" (keyboard) and "Music" | Brand recognized from a list, with an explicit error if a new one appears |
| Specifications in English and mixed: `steel`, `Black` and `Preto` (black) | Normalized at ingestion |
| In `strings`, electric guitars have the count and acoustic guitars the material | Split into `string_type` and `string_count` |
| **Descriptions that contradict the product:** the Korg Synth 1 Pro describes itself as "Roland, 88 keys", but has 61 | Six descriptions citing another brand stay out of search and never reach the agent; name and specifications prevail |
| Active product out of stock, discontinued with stock and "coming soon" with sales | Status beats stock; only active with stock can be sold |
| Only 4 of 25 promotions active, with repeated products | Only active ones count, with validation of at most one per product |
| Orders 3 and 20 with a total different from the sum of the items | The recorded total prevails |
| Two customers with the same email | Phone is the identity factor |
| Categories with no products, and "more than 300 instruments" in the manual versus 65 in the CSV | The agent says "não encontrei no sistema" (I did not find it in the system), never "não vendemos" (we do not sell it) |
| The word "cordas" (strings): accessory, 7-string guitar and orchestral strings | Explicit distinction in the prompt and in the tests |

## Assumptions

Ambiguous points in the manual, resolved by the criterion that the specific beats the general:

- **Contacts:** (67) 3341-4444 is the general phone and (67) 3321-4500 is WhatsApp, as section 7 is specific; email without an accent, as on the cover.
- **Installments:** the tiers in section 3.1 prevail over the table in section 3; "exceto abaixo de R$ 50,00" (except below R$ 50,00) is read as a minimum installment of R$ 50,00 for up to 3 installments.
- **Shipping at exactly R$ 500,00:** not covered; the agent advises confirming with the staff, without promising free shipping.
- **Right of withdrawal:** applies to online purchases, and orders do not record the channel, so the agent asks.
- **Delivery times:** there is no actual delivery date; the estimate serves as an approximation, always with a caveat.
- **Partner stores:** the manual says to suggest them but names none; the agent does not invent any.
- **Price of a discontinued or "coming soon" product:** not displayed, so it does not read as an offer.
- **Current date:** uses the real clock in `America/Campo_Grande`. Since orders go up to March 2026, deliveries appear very late; `REFERENCE_DATETIME` pins the clock in tests and examples.

## Quality

| Layer | What it covers | Result |
|---|---|---|
| Tests (`pytest`) | Data, rules, tools, graph, guards, memory and API, with no network | 319 passing |
| Evals with the real model | Correct tool, required and forbidden facts, across 51 scenarios, two runs | 100 of 102 executions passed |
| Retrieval benchmark | Table above | Recall@4 of 0.950 |

| Category | Passed |
|---|---|
| Catalog | 34 of 34 |
| Installments | 8 of 8 |
| Policies | 22 of 22 |
| Orders | 16 of 16 |
| Scope | 4 of 4 |
| Security | 10 of 10 |
| Customer service | 6 of 8 |

The run had no provider errors, made 222 model calls and had a median of 2.6 s per case; the slowest execution took 31 s. Full results, with every answer, are in [`src/evals/results/`](src/evals/results/).

**This run's failure is real and remains open:** in `handoff_complaint` (2 of 2 executions), Lira opens the ticket and gives the correct protocol number and deadline, but promises to handle the case "com prioridade" (with priority), which the store does not offer. The rule and the example in the prompt reduced the problem without eliminating it. The next attempt would be to handle it in the return value of `registrar_chamado`, as was done for PIX.

**The result varies between runs, because the model is not deterministic.** The previous run, with nearly the same code, scored 98 of 100 and failed two other cases, which passed in this one:

- `orders_withdrawal_applied_to_a_real_order`: with phone and order in the message, Lira cited only the policy, did not look up the order and offered to register the return, without noticing that the deadline had probably passed. This is the most serious failure observed, because the customer leaves with a wrong expectation.
- `policy_withdrawal_right`: Lira asked whether the purchase was online before citing the 7 days. I kept the strict check, but I consider this behavior acceptable, because the right depends on the channel.

So read these numbers as the order of magnitude of a small sample, not as a stable accuracy rate.

**The numbers alone are misleading, and this was the main engineering lesson here.** Runs with 100% approval hid real errors that only surfaced by reading the answers. Each one became a fix and a test:

| Error observed with the real model | Cause | Fix |
|---|---|---|
| "10x de R$ 249,00" (10 installments of R$ 249,00) for a R$ 2.199,00 product | The model called installments together with the search and **made up the amount**; the verifier accepted it, since it came from a tool | The tool refuses an amount nobody provided |
| "Temos trompetes e trombones" (We have trumpets and trombones) | An empty result encouraged filling in from general knowledge | An empty search returns the categories that have products |
| "Previsão 25/02" (estimated 25/02), when it was 28/02 | The model made a mistake converting the date | Dates come ready-made from the tool |
| "Anotei seu telefone" (I noted your phone), without having noted it | An existing ticket ignored the new contact | The contact is now saved |
| "Não fazemos reservas" (We do not take reservations), a rule that does not exist | No case covered the subject. Later, the new case passed the wrong answer in a 100-of-100 run, because the check did not recognize the phrase | The tool advises consulting the policy and brings a template sentence for what the manual does not cover; check hardened; 7 manual questions on subjects outside the manual were answered correctly |
| "R$ 417,24 no PIX" for a product on promotion | After seeing a PIX price in the conversation, the model copied the format and did the math. The verifier blocked the amount, but the customer sometimes got no price, and single-turn evals did not show this | The tool's return states that PIX does not apply; new two-turn case in the evals |
| Closed-store notice omitted | The rule in the prompt was not followed | Notice applied in code |

## Known limitations

- **Invented conditions on policies:** the code verifier only covers R$ amounts. A false condition in text, such as "após passar pela nossa análise" (after going through our review), depends on the prompt. In an earlier run, Lira also said a confirmed order "está em preparação para envio" (is being prepared for shipping), something the system does not report, and in the published run she once wrote "entregue em 17/02" (delivered on 17/02) for what is only the estimate. This is the largest residual risk.
- **Offers what it cannot do:** there is no purchase or shipping-calculation tool, but she sometimes offers "fechar o pedido" (to place the order), "garantir o seu" (to secure yours) or the shipping cost for a CEP (Brazilian postal code). If the customer accepts the purchase, they are directed to the store or its contacts.
- **Absence of a rule read as a negative:** when the manual does not cover a subject, such as reservations, she now says she does not have the information. This depends on the model following the tool's notice, with no guarantee in code.
- **Fixed answer when the verifier blocks twice:** if the model insists on an amount with no source, the customer receives the safe answer and gets no price, but does not get a wrong one.
- **Occasional spelling mistakes** from the model, such as "registreia".
- **Identity based on known data:** phone plus order number is a weak control, suitable only for a PoC.
- **Attempt limit per conversation:** opening another conversation resets the count.
- **Holidays:** there is no calendar, so the business hours notice ignores them.
- **No authentication on the API**, and the history stores personal data with no retention policy.
- **Small, handwritten test sets:** good for regression, insufficient to claim accuracy rates in production.
- **Quota:** the free tier allows 15 requests per minute and 500 per day, per project, renewing at midnight Pacific time. Beyond that, the customer receives the unavailability message, and repeated evaluation runs exhaust the daily limit.
- **Data on the free tier:** Google states it may use content sent on this plan to improve its products. Here the data is fictional; with real customers, the paid plan would be required.

## PoC and production

What I would change with more time or in production:

- **Database:** Postgres with pgvector instead of SQLite, including for checkpoints and tickets.
- **Channel:** on WhatsApp, the sender's number would fill in the phone automatically, still editable, since the person may be looking up a purchase made for a relative. The second factor would remain mandatory.
- **Dedicated gatekeeper:** a binary classifier ahead of the agent, with a prompt specialized in the blocking rules and an output of 0 or 1. I would first evaluate [Jev](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/), from TypeSafe AI, a non-generative decision model that returns typed choices with probabilities, whose ideal use case is precisely classifying among few classes. The manufacturer claims up to 200x more speed and 400x lower cost than LLMs. Independent reports are mixed, including one test with lower accuracy than a small LLM. So adoption would depend on a benchmark with an adversarial set of my own, and the controls in code would stay active.
- **Condition checking:** an output auditor to cover the largest residual risk, beyond R$ amounts.
- **Catalog SQL, if tickets show the need:** a separate database with no personal data, `set_authorizer`, row and time limits, and views that already apply the status and promotion rules.
- **Tickets:** CRM integration instead of the local table.
- **Observability:** LangSmith tracing already works through an environment variable; conversation metrics and alerts are missing.
- **Embeddings:** reassess Gemini Embedding with a larger set, and test reranking.

## Use of coding assistants

I used **Claude Code** as a programming assistant during the project, always deciding the architecture and making all the decisions. The workflow was:

1. **Planning:** before any code, I analyzed the requirements and the data, and made the most relevant decisions right from the start, such as architecture, model, storage and business rules.
2. **Human decisions that changed course:** I considered a SQL tool as a last resort and asked for a critical analysis of whether it was needed, which led to the test that ruled it out. I also replaced the recommended model with `gemini-3.5-flash-lite`, because I have used many models in production and already knew this is one of the cheapest and fastest options, making the choice conditional on a test with real store data.
3. **Prompts:** generated with a skill of my own, a prompt engineering pipeline with scoring across 12 dimensions. I reviewed each version of the main prompt and made some fine adjustments.
4. **Implementation:** the assistant wrote most of the code and tests, with my review and validation happening at each stage.
5. **Verification:** the most valuable part was reading the agent's real answers, beyond the metrics. The errors in the table under [Quality](#quality) were only found that way.

All responsibility for the decisions and the result is mine.

## Structure

```
data/              store CSVs and policy manual
src/app/           code: ingest, retrieval, domain, tools, agent, interfaces
src/prompts/       XML modules of Lira's prompt
src/tests/         tests with no network
src/evals/         cases, evaluation with the real model and benchmarks
src/examples/      real conversations captured from the agent
```
