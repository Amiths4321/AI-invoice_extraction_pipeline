"""
Configuration for the invoice extraction pipeline.

All values can be overridden via environment variables (or a .env file —
see .env.example). The pipeline talks to a self-hosted Qwen model served
behind an OpenAI-compatible /v1/chat/completions endpoint (this is what
vLLM, TGI, LM Studio, and Ollama's compatibility mode all expose), over
plain HTTP — no third-party SDK involved.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; if it's not installed, env vars must be
    # set some other way (shell export, Docker env, CI secrets, etc.)
    pass

BASE_DIR = Path(__file__).resolve().parent

# Base URL of your remote Qwen server, WITHOUT a trailing slash and
# WITHOUT "/v1/chat/completions" (that suffix is appended in extractor.py).
# Examples:
#   vLLM:    http://your-server:8000
#   Ollama:  http://your-server:11434
#   TGI:     http://your-server:8080
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "http://localhost:8000")

# The exact model identifier your server expects. This MUST match what
# the server was launched/pulled with — e.g. the vLLM --served-model-name
# value, or the Ollama tag (such as "qwen2.5:32b-instruct"). There's no
# safe universal default, so set this explicitly in .env.
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5-instruct")

# Optional. Many self-hosted setups have no auth at all; leave blank if so.
# If set, sent as "Authorization: Bearer <token>".
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

# Line items can be long for multi-page invoices; give headroom.
MAX_TOKENS = int(os.getenv("INVOICE_PIPELINE_MAX_TOKENS", "4096"))

# Deterministic extraction — we want the same invoice to produce the same
# JSON every time, not creative variation.
TEMPERATURE = float(os.getenv("INVOICE_PIPELINE_TEMPERATURE", "0"))

# Per-request timeout. Self-hosted models on modest hardware can be slow,
# especially for longer invoices — default is generous on purpose.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("INVOICE_PIPELINE_TIMEOUT", "120"))

# Retry behaviour for transient errors (connection drops, timeouts, 5xx).
MAX_RETRIES = int(os.getenv("INVOICE_PIPELINE_MAX_RETRIES", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.getenv("INVOICE_PIPELINE_RETRY_DELAY", "2"))

# Path to the static system prompt file.
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "system_prompt.txt"

# Concurrency cap for batch CLI runs.
BATCH_MAX_WORKERS = int(os.getenv("INVOICE_PIPELINE_BATCH_WORKERS", "4"))
