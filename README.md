# AI Basic Inventory Agent Loop

A local inventory assistant for Carla's coffee supply stores: a FastAPI API backed
by CSV and a terminal agent using NVIDIA or Groq through the OpenAI Python client. The agent
loop is implemented manually, without an agent framework.

## Setup

Requires Python 3.10+ and a NVIDIA or Groq API key with access to a tool-capable model.
Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

For NVIDIA, replace the Groq settings in `.env` with:

```dotenv
NVIDIA_API_KEY=your_key_here
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```

Replace the key placeholder privately. Use a tool-capable model available to your
NVIDIA account; set `LLM_MODEL` to its exact model ID if different. NVIDIA defaults
are selected automatically when `NVIDIA_API_KEY` is present. Existing
`LLM_BASE_URL` and `LLM_MODEL` settings override those defaults, so remove or
replace any old Groq URL/model values when switching providers.

For Groq instead, set `GROQ_API_KEY` and leave `NVIDIA_API_KEY` unset. Never commit your key: `.env`, local
inventory, and conversation logs are already excluded by `.gitignore`.
If the default model is unavailable for your account, set `GROQ_MODEL` to another
Groq model that supports tool calling.

## Start And Stop

Use two terminals, with the virtual environment activated in each.

Terminal 1, start the API first:

```bash
uvicorn api.app:app --reload
```

Terminal 2:

```bash
python agent.py
```

The API is at http://127.0.0.1:8000 and interactive API docs are at
http://127.0.0.1:8000/docs. The agent checks that the API is available at startup.
Type `exit` or `quit`, or press Ctrl+C, to stop the agent. Then press Ctrl+C in
the API terminal. To start another conversation, run `python agent.py` again;
the API can remain running.

## Example Conversation

The initial catalog is empty. Register products through the agent or API:

```text
You: Register Oat milk at Downtown with 8 liters, and check low-stock alerts.
You: Register Oat milk at Uptown with 20 liters.
You: We just received 30 liters of oat milk at Downtown.
You: Register Arabica at Downtown with 40 bags.
You: We sold 12 bags of Arabica at Downtown today. What products are running low?
You: Do we have enough Arabica at Downtown for the week if we expect to sell 35 bags?
```

Each location has its own product ID. Location defaults to `Main` when omitted.
Use consistent location names and units; the agent asks for clarification when
a product or location is ambiguous. Alerts compare each product's numeric
quantity to the threshold in its own unit. They are not demand forecasts:
weekly coverage needs an expected usage figure.

## REST API

| Method | Endpoint | Behavior |
| --- | --- | --- |
| GET | `/inventory` | List products; optional `location` query filter. |
| POST | `/inventory` | Register `name`, nonnegative `quantity`, `unit`, optional `location`; returns 201. |
| PATCH | `/inventory/{product_id}` | Apply `{"delta": 30}` for deliveries or `{"delta": -12}` for sales. |
| GET | `/inventory/alerts` | Products strictly below `threshold`; optional `location` filter. |

Example product body:

```json
{"name": "Oat milk", "quantity": 8, "unit": "liters", "location": "Downtown"}
```

Successful reads and updates return 200. Unknown IDs return 404; duplicate names
at the same location and insufficient stock return 409; invalid input returns
422. Errors contain a descriptive `detail`. Stock cannot become negative, and
quantities and deltas must be finite. Names are case-insensitive for duplicate
detection within each location.

`LOW_STOCK_THRESHOLD` sets the API default (10); restart the API after changing
it. A request can override it, for example `/inventory/alerts?threshold=15`.

## Persistence And Agent Loop

`products.csv` is created automatically and has fields
`id,name,quantity,unit,location`. Atomic replacements and file locks protect
updates, including concurrent requests. Products survive API restarts. Back up
the CSV; do not edit it while the API is running. Invalid CSV data is rejected
rather than silently overwritten.

The agent follows Observe -> Think -> Act -> Update -> Repeat. It sends the
full session history and four typed tools to the model, executes selected tools
over HTTP, then appends each result with its tool-call ID before asking the
model again. Multiple tool calls and multiple rounds are supported. The loop
ends when the model responds without tool calls; a 12-round safety limit
prevents unbounded execution.

Every user message, assistant message, tool call, and tool result is immediately
appended to `conversation_log.csv` with fields
`actor,message,tool_call,timestamp`. Actors are `user`, `agent`, and `tool`;
timestamps are timezone-aware ISO 8601. Tool calls are logged as agent events
with their name and arguments; results are tool events. The file is flushed
after each event, and a new session never overwrites old rows. Session context
starts fresh in memory; past CSV logs are retained but not replayed to the LLM.

Optional `.env` settings include `INVENTORY_API_URL`, `LLM_BASE_URL`, `LLM_MODEL`,
`PRODUCTS_FILE`, and `CONVERSATION_LOG_FILE`. File paths default to the repository
root; relative overrides are resolved from the process working directory.

Network and API errors are returned to the model instead of reported as success.
If a connection fails during a change, inspect inventory before retrying:
the server may have saved the change even if the response was lost.

This is a local prototype with no authentication. Keep the API on localhost.
Inventory/tool results and messages are sent to the configured LLM provider;
keep personal information and secrets out of conversations and protect local logs.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary CSV files and a scripted model, so no API key or running
server is needed. They cover persistence, stock validation, fractional amounts,
concurrent updates, all tool routes, multi-step interactions, result injection,
session history, append-only logs, provider selection, and error handling. A real LLM conversation
requires your own key and network access.
