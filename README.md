# Laya AI quick demo

A five-minute tour of [Laya](https://github.com/NandhaKishorM/laya), the open-source
System 1 decision model from Convai Innovations (Apache 2.0, released September 2026).

Laya is not a chat model. You give it a *state* (free text or a JSON-like record) and a
set of *typed questions*, and it answers all of them in one forward pass with calibrated
probabilities. No text is generated, so there is nothing to parse and nothing to
hallucinate. It is built for the boring, high-volume decisions that sit in front of
bigger systems: which queue gets this ticket, how urgent is it, is this condition true.

This repo triages a handful of support tickets, in English, Spanish, Hindi, German and
Chinese, to show the three question types and the language router.

## What the demo shows

| Question type | Example in `demo.py` | What comes back |
| --- | --- | --- |
| `choice` | which department handles the ticket | one label plus a calibrated `confidence` |
| `score` | how urgent it is on a 3-level scale | a real number on `0 .. levels-1` |
| `noul` | does the user threaten to churn, ask for a refund | probability of "yes" |

Every answer also carries `routing` metadata: which checkpoint ran (`english` or
`multilingual`) and the reason, decided in under a millisecond from the script and
language of the input before any forward pass.

## Run it

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). The first run downloads two
checkpoints (about 1.4 GB total) from Hugging Face into `~/.cache/huggingface`; later
runs load them from disk in a few seconds.

```bash
make setup        # uv sync: creates .venv, installs laya==0.3.22
make route        # routing decisions only, instant, no model download
make demo         # triage the sample tickets
uv run python demo.py "My card was charged twice, fix it today or I'm leaving"
```

`demo.py` is short and self-contained: a question schema, a list of tickets, and a loop.
Read it top to bottom; the comments explain each question type.

## What you should see

Output from an Apple-silicon Mac (PyTorch on MPS), trimmed:

```text
  english       ready in   5.0 s
  multilingual  ready in   2.5 s
> We were billed twice for March. Refund the duplicate today or we cancel our plan.
  checkpoint : english  (English Latin text)
  department : billing  (confidence 0.78)
  urgency    : 1.36 / 2
  churn risk : 68%
  refund?    : 88%
  latency    : 136 ms for 4 questions
  -> auto-route to billing
> La aplicación se cierra cada vez que abro la configuración. No es urgente, pero es mo...
  checkpoint : multilingual  (Latin script but language looks like 'es', not English)
  department : technical  (confidence 0.99)
  urgency    : 1.81 / 2
  churn risk : 1%
  refund?    : 0%
  latency    : 50 ms for 4 questions
  -> auto-route to technical
> मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।
  checkpoint : multilingual  (non-Latin script (devanagari, 100% of letters); ...)
  department : sales  (confidence 0.29)
  urgency    : 1.65 / 2
  churn risk : 1%
  refund?    : 98%
  latency    : 50 ms for 4 questions
  -> escalate to a human (confidence 0.29 < 0.7)
```

Things worth pointing out when you walk someone through it:

- Four questions, one forward pass. On this laptop the English checkpoint
  (ModernBERT-large) takes 100 to 135 ms per ticket and the multilingual one
  (mmBERT-base) 35 to 50 ms. The authors' 33 ms figure is a T4 GPU.
- The router never touches the model. Script and language detection run in pure Python
  before inference, and the `reason` string tells you why a checkpoint was picked.
- Confidence is doing real work. The Hindi ticket is a refund request, and the model got
  `refund? 98%` right but picked the wrong department at 0.29 confidence. That low number
  is exactly what the gate is for: it goes to a human instead of the sales queue.
- Zero-shot has misses. The German ticket announces a cancellation and `churn_risk` still
  came back at 1% from the multilingual checkpoint. The same question on English text
  scored 68% and 76%. This is why the authors push fine-tuning on your own decisions.
- `score` answers are real numbers, not labels. `1.81 / 2` on a three-level scale means
  "between soon and critical, closer to critical"; round or bucket it yourself.

## Where it fits

Laya is a good fit when you have many small, bounded decisions and want them fast, cheap
and auditable: ticket routing, intent classification, urgency scoring, yes/no policy
gates, confidence-gated hand-off to a human or to an LLM. It is not a fit when the answer
is open-ended text, or when one choice question has dozens of labels (the authors call
this out as a known weak spot).

## Going further

- `Router.predict_batch(...)` packs many states into shared forward passes for backlogs.
- `laya "some text" --preset triage` runs a ready-made question set from the CLI.
- `make serve` starts the HTTP server (`laya[serve]`) on `http://127.0.0.1:8000`; the
  `POST /v1/systemone` body is the same `{state, questions}` shape used here.
- Zero-shot accuracy is the floor. The authors report a jump from 0.36 to 0.77 on their
  typed-decisions benchmark after fine-tuning on in-domain decisions; the notebook runs on
  free Kaggle GPUs.
- Docs: <https://nandhakishorm.github.io/laya/> · model card:
  <https://huggingface.co/convaiinnovations/laya>
