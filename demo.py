"""Quick demo of Laya: typed decisions over support tickets, in several languages.

Laya is not a chat model. You hand it a *state* (text or a JSON-like dict) and a set
of *typed questions*; it answers all of them in one forward pass (~30-40 ms on a GPU)
and returns calibrated probabilities instead of generated text. Nothing to parse,
nothing to hallucinate.

This script triages a handful of support tickets to show the three question types
(choice, score, noul) and the language router that picks a checkpoint per request.

Run:
    uv run python demo.py                                  # built-in sample tickets
    uv run python demo.py "My card was charged twice"      # your own ticket text
    uv run python demo.py --route-only                     # routing decisions only, no model load
"""

from __future__ import annotations

import sys
import time
import warnings

from laya import Router

# The English checkpoint warns that its calibration temperatures for choice questions
# with 11+ options are invalid. Our choice question has 4 options, so it does not apply.
warnings.filterwarnings("ignore", message=r".*invalid temperatures.*", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# 1. The question schema. Laya answers every question below in a single pass.
# ---------------------------------------------------------------------------
QUESTIONS = {
    # choice: pick exactly one label. The criteria describe what each label means.
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
            "billing": "invoices, payments, refunds, duplicate charges",
            "technical": "bugs, crashes, outages, login or system errors",
            "sales": "pricing, upgrades, new contracts, quotes",
            "other": "everything else",
        },
    },
    # score: an ordinal scale. The criteria are the levels from low to high,
    # and the answer is a real number on that scale (0 .. len(criteria) - 1).
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    # noul: a yes/no question, answered as the probability of "yes".
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or leave?",
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly ask for a refund?",
    },
}

# ---------------------------------------------------------------------------
# 2. Sample tickets. Mixed languages so the Router has something to route.
#    A state can be a plain string or a dict; dicts are read field by field.
# ---------------------------------------------------------------------------
TICKETS: list[dict | str] = [
    {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "We were billed twice for March. Refund the duplicate today or we cancel our plan.",
    },
    {
        "from": "dev@startup.io",
        "subject": "500 errors since this morning",
        "body": "Every API call returns HTTP 500 since 09:00 UTC. Production is down for all our customers.",
    },
    {
        "from": "cto@bigco.com",
        "subject": "Enterprise pricing",
        "body": "We're evaluating the enterprise tier for 400 seats. Can someone send a quote for next quarter?",
    },
    "La aplicación se cierra cada vez que abro la configuración. No es urgente, pero es molesto.",
    "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
    "Ich bin sehr unzufrieden und werde zum Monatsende kündigen, wenn der Login nicht heute repariert wird.",
    "請問你們的辦公時間是幾點？",
]

# Confidence below which a decision goes to a human instead of being acted on.
# Laya's probabilities are calibrated (trained against proper scoring rules), so a
# threshold is meaningful. 0.70 is illustrative: fit it on your own held-out data.
AUTO_ROUTE_THRESHOLD = 0.70

URGENCY_MAX = len(QUESTIONS["urgency"]["criteria"]) - 1


def preview(ticket: dict | str, width: int = 88) -> str:
    text = ticket if isinstance(ticket, str) else ticket.get("body", str(ticket))
    return text if len(text) <= width else text[: width - 3] + "..."


def show_routing(router: Router, tickets: list[dict | str]) -> None:
    """Language detection only: sub-millisecond, pure Python, no checkpoint needed."""
    print("Routing decisions (no forward pass):\n")
    for ticket in tickets:
        decision = router.route(ticket, QUESTIONS)
        print(f"  {decision.model:<13} {preview(ticket, 60)}")
        print(f"  {'':<13} {decision.reason}\n")


def warm_up(router: Router) -> None:
    """Load both routed checkpoints and run the real schema once through each, so the
    per-ticket timings below measure inference rather than loading and first-call setup."""
    for model in ("english", "multilingual"):
        started = time.perf_counter()
        router.predict("Hello, I have a question about my account.", QUESTIONS, model=model)
        print(f"  {model:<13} ready in {time.perf_counter() - started:5.1f} s")


def triage(router: Router, ticket: dict | str) -> None:
    started = time.perf_counter()
    result = router.predict(ticket, QUESTIONS)
    elapsed_ms = (time.perf_counter() - started) * 1000

    answers = result["answers"]
    routing = result["routing"]
    dept = answers["department"]

    print(f"\n> {preview(ticket)}")
    print(f"  checkpoint : {routing['model']}  ({routing['reason']})")
    print(f"  department : {dept['choice']}  (confidence {dept['confidence']:.2f})")
    print(f"  urgency    : {answers['urgency']['score']:.2f} / {URGENCY_MAX}")
    print(f"  churn risk : {answers['churn_risk']['noul']:.0%}")
    print(f"  refund?    : {answers['refund_requested']['noul']:.0%}")
    print(f"  latency    : {elapsed_ms:.0f} ms for {len(QUESTIONS)} questions")

    if dept["confidence"] >= AUTO_ROUTE_THRESHOLD:
        print(f"  -> auto-route to {dept['choice']}")
    else:
        print(f"  -> escalate to a human (confidence {dept['confidence']:.2f} < {AUTO_ROUTE_THRESHOLD})")


def main(argv: list[str]) -> None:
    router = Router()  # lazy: downloads and loads a checkpoint the first time it is needed

    if argv and argv[0] == "--route-only":
        show_routing(router, TICKETS)
        return

    tickets: list[dict | str] = [" ".join(argv)] if argv else TICKETS

    print("Loading checkpoints (first run downloads them from Hugging Face)...")
    warm_up(router)

    for ticket in tickets:
        triage(router, ticket)


if __name__ == "__main__":
    main(sys.argv[1:])
