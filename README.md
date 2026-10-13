# DecisionUI

**Turn any LLM into a decision model.**

DecisionUI is a lightweight Python server that sits in front of a Chat Completions endpoint (vLLM, SGLang, Ollama, a hosted API, etc.) and exposes it through two decision APIs:

- **SystemOne API**
- **OpenAI Decisions API**

Instead of parsing free-form text, you ask structured questions and get back the chosen option along with its probabilities.

## Features

- **Drop-in**: works with any OpenAI-compatible Chat Completions server, with no changes to the model or the server.
- **Structured decisions**: define choices and criteria, and get back typed results.
- **Probabilities included**: inspect the model's confidence for every option, not just the top pick.
- **Two APIs, one server**: SystemOne and OpenAI Decisions endpoints are served side by side.

## Installation

```bash
pip install decision-ui
```

## Quick Start

**1. Serve a model** with any OpenAI-compatible inference engine, for example vLLM:

```bash
vllm serve google/gemma-4-31B-it --host localhost --port 8000
```

**2. Launch DecisionUI** and point it at the model server:

```bash
decision-ui serve http://localhost:8000 --host localhost --port 9000
```

**3. Make a decision**

## Usage

```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(base_url="http://localhost:9000", api_key="EMPTY") as client:
    decision = client.system_one(
        model="google/gemma-4-31B-it",
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "category": Choice(
                instructions="What is this ticket about?",
                criteria={"billing": None, "technical": None, "other": None},
            ),
        },
    )

print(decision.choices["category"].probabilities)
```

Example output:

```python
{"billing": 0.94, "technical": 0.03, "other": 0.03}
```

### How it works

```
Your app ──► DecisionUI (:9000) ──► Chat Completions server (:8000) ──► LLM
          SystemOne / Decisions API        e.g. vLLM
```

## CLI Reference

```bash
decision-ui serve <backend_url> [--host HOST] [--port PORT]
```

| Option        | Description                                      | Default     |
|---------------|--------------------------------------------------|-------------|
| `backend_url` | Base URL of the OpenAI-compatible model server.  | required    |
| `--host`      | Interface DecisionUI binds to.                   | `localhost` |
| `--port`      | Port DecisionUI listens on.                      | `9000`      |