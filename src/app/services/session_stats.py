"""Session (thread) context + usage stats, computed from stored messages.

Best-practice model: token usage is read from the standard langchain
`usage_metadata` attached to finalized AI messages (consistent across
providers), and the session view distinguishes two numbers:

- **Cumulative usage** (cost tracking): `input_tokens`/`output_tokens`
  summed over every run. Input is *billed* input — each run's input includes
  the history, so summing counts tokens more than once; output is additive
  and exact.
- **Current context** (window management): the `input_tokens` of the most
  recent run is exactly the prompt the model saw last, i.e. the context it
  currently holds (OpenAI/Anthropic context-monitoring practice). Comparing
  it to the model's context window gives utilization + remaining tokens.

The context window table is a curated best-effort heuristic keyed by model
id prefix (first match wins); unknown models report `None` instead of a
guessed number. The pricing table follows the same pattern (USD per 1M
input/output tokens) and powers the thread's estimated API cost.
"""

from __future__ import annotations

from typing import Any

# (model id prefix, context window in tokens) — first match wins, longest
# specific prefixes before generic ones. Sourced from provider docs (Oct 2025).
_CONTEXT_WINDOW_RULES: list[tuple[str, int]] = [
    # OpenAI
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4-turbo", 128_000),
    ("gpt-4", 8_192),
    ("o4-mini", 200_000),
    ("o3-mini", 200_000),
    ("o3", 200_000),
    ("o1", 200_000),
    ("gpt-3.5-turbo", 16_385),
    # Anthropic
    ("claude", 200_000),
    # Google
    ("gemini", 1_048_576),
    # DeepSeek
    ("deepseek-chat", 128_000),
    ("deepseek-reasoner", 128_000),
    # Meta
    ("llama-3.1", 131_072),
    ("llama-3.3", 131_072),
    ("llama-3.2", 131_072),
    # Mistral
    ("mistral-large", 128_000),
    ("mistral-small", 128_000),
    ("mistral-medium", 32_000),
    # Cohere
    ("command", 256_000),
    # xAI
    ("grok", 131_072),
]

# (model id prefix, input USD per 1M tokens, output USD per 1M tokens) —
# first match wins, specific prefixes before generic ones. Sourced from
# provider pricing pages (Oct 2025). Models with provider-dependent pricing
# (llama, mistral, grok, command) are intentionally absent -> cost is None.
_PRICING_RULES: list[tuple[str, float, float]] = [
    # OpenAI
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4o", 2.50, 10.00),
    ("gpt-4.1", 2.00, 8.00),
    ("gpt-4-turbo", 10.00, 30.00),
    ("gpt-4", 30.00, 60.00),
    ("o4-mini", 1.10, 4.40),
    ("o3-mini", 1.10, 4.40),
    ("o3", 2.00, 8.00),
    ("o1", 15.00, 60.00),
    ("gpt-3.5-turbo", 0.50, 1.50),
    # Anthropic
    ("claude-opus", 15.00, 75.00),
    ("claude-sonnet", 3.00, 15.00),
    ("claude", 3.00, 15.00),
    # Google
    ("gemini-2.5-pro", 1.25, 10.00),
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini", 1.25, 10.00),
    # DeepSeek
    ("deepseek-chat", 0.27, 1.10),
    ("deepseek-reasoner", 0.55, 2.19),
]

_USAGE_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def _usage_of(message: Any) -> dict | None:
    """The message's usage_metadata dict, or None when it has none."""
    usage = message.get("usage_metadata") if isinstance(message, dict) else None
    return usage if isinstance(usage, dict) else None


def compute_usage(messages: list[dict]) -> dict | None:
    """Cumulative token usage over AI messages that report usage_metadata.

    Returns {"input_tokens", "output_tokens", "total_tokens", "runs"} or None
    when no message reports usage (scripted/local models).
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    runs = 0
    for m in messages:
        usage = _usage_of(m)
        if usage is None or not any(isinstance(usage.get(k), int) for k in _USAGE_KEYS):
            continue
        runs += 1
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    if not runs:
        return None
    return {**totals, "runs": runs}


def current_context_input_tokens(messages: list[dict]) -> int | None:
    """Input tokens of the most recent run — the context the model currently sees.

    Scans backwards for the latest message with usage_metadata (the final AI
    message of the last run; `input_tokens` covers the whole prompt incl.
    history). None before the first run reports usage.
    """
    for m in reversed(messages):
        usage = _usage_of(m)
        if usage is not None and isinstance(usage.get("input_tokens"), int):
            return usage["input_tokens"]
    return None


def context_window_for(model: str | None) -> int | None:
    """Best-effort context window for a `provider:model` string (None = unknown)."""
    model_id = _model_id(model)
    if model_id is None:
        return None
    for prefix, window in _CONTEXT_WINDOW_RULES:
        if model_id.startswith(prefix):
            return window
    return None


def _model_id(model: str | None) -> str | None:
    """Lowercased model id from a `provider:model` string, or None."""
    if not model:
        return None
    return (model.partition(":")[2] or model).lower()


def pricing_for(model: str | None) -> dict | None:
    """Best-effort per-1M-token rates for a `provider:model` string (USD).

    Returns {"input_per_million", "output_per_million"} or None when the
    model's pricing is unknown (provider-dependent models are absent from
    the curated table).
    """
    model_id = _model_id(model)
    if model_id is None:
        return None
    for prefix, input_price, output_price in _PRICING_RULES:
        if model_id.startswith(prefix):
            return {"input_per_million": input_price, "output_per_million": output_price}
    return None


def estimate_cost(usage: dict | None, model: str | None) -> dict | None:
    """Estimated API cost (USD) of the thread's cumulative `usage`.

    Billed at the model's per-1M-token rates; None when the model's pricing
    is unknown or no usage was reported. Note `input_tokens` is billed input
    (history counted per run), so the estimate matches what the provider
    would charge for the conversation so far.
    """
    if not usage:
        return None
    pricing = pricing_for(model)
    if pricing is None:
        return None
    input_tokens = usage.get("input_tokens") or 0
    output_tokens = usage.get("output_tokens") or 0
    input_cost = round(input_tokens / 1_000_000 * pricing["input_per_million"], 8)
    output_cost = round(output_tokens / 1_000_000 * pricing["output_per_million"], 8)
    return {
        "currency": "USD",
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": round(input_cost + output_cost, 8),
        "pricing": pricing,
    }


def message_counts(messages: list[dict]) -> dict:
    """{"count": messages, "characters": total content length}."""
    return {
        "count": len(messages),
        "characters": sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict)),
    }


def build_context(current_input: int | None, model: str | None) -> dict | None:
    """The context report for a thread; None when no run has reported usage."""
    if current_input is None:
        return None
    window = context_window_for(model)
    return {
        "current_input_tokens": current_input,
        "context_window": window,
        "utilization": round(current_input / window, 4) if window else None,
        "remaining_tokens": window - current_input if window is not None else None,
    }


__all__ = [
    "build_context",
    "compute_usage",
    "context_window_for",
    "current_context_input_tokens",
    "estimate_cost",
    "message_counts",
    "pricing_for",
]
