# DecisionUI
A decision interface server, compatible with SystemOne API and OpenAI's Decisions API.

## Usage
Install DecisionUI
```bash
pip install decision-ui
```

Deploy the model using any inference engine, e.g., vLLM.
```bash
vllm serve google/gemma-4-31B-it --host localhost --port 8000
```

Start a DecisionUI server
```bash
decision-ui serve --base_url http://localhost:8000 --port 8080
```

Use it via TypeSafeClient
```python
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(base_url="http://localhost:8080/v1", api_key="EMPTY") as client:
    decision = client.system_one(
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