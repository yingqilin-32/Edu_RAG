"""Per-request model limits, metered at the HTTP boundary, including planner calls."""

from time import perf_counter
from json import dumps

import httpx

from edu_rag.llm import ModelError


class BudgetExceeded(ModelError):
    pass


class BudgetClient:
    def __init__(self, *, max_calls, max_tokens, max_seconds, timeout, client=None):
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.started, self.max_seconds = perf_counter(), max_seconds
        self.stage, self.records, self.denied = "understand", [], None
        self.timeout, self.owned = timeout, client is None
        self.client = client if client is not None else httpx.Client(follow_redirects=False)

    @property
    def spent_tokens(self):
        return sum(r["usage"].get("total_tokens") if isinstance(r["usage"].get("total_tokens"), int)
                   else r["input_and_output_estimate"] for r in self.records)

    def reject(self, reason):
        self.denied = reason
        raise BudgetExceeded(reason)

    def post(self, url, *, headers, json):
        remaining = self.max_seconds-(perf_counter()-self.started)
        if len(self.records) >= self.max_calls:
            self.reject("model_call_limit")
        if remaining <= 0:
            self.reject("model_deadline")
        # Conservative admission estimate, NOT a provider-tokenizer guarantee.
        estimated = len(dumps(json["messages"], ensure_ascii=False).encode("utf-8"))+256+json["max_tokens"]
        if self.spent_tokens+estimated > self.max_tokens:
            self.reject("token_admission_limit")
        record = {"stage": self.stage, "usage": {}, "status": "error", "input_and_output_estimate": estimated,
                  "output_cap": json["max_tokens"]}
        self.records.append(record)
        start = perf_counter()
        try:
            kwargs = {"headers": headers, "json": json}
            if isinstance(self.client, httpx.Client):
                kwargs["timeout"] = httpx.Timeout(min(remaining, self.timeout), connect=min(10, remaining))
            response = self.client.post(url, **kwargs)
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage", {})
            record.update(status="ok", usage=usage if isinstance(usage, dict) else {}, response_model=payload.get("model"))
            if self.spent_tokens > self.max_tokens:
                self.reject("actual_token_limit")
            return response
        finally:
            record["elapsed_ms"] = round((perf_counter()-start)*1000, 3)

    def can_retry(self, reserve_final_calls):
        return len(self.records)+3+reserve_final_calls <= self.max_calls and self.spent_tokens < self.max_tokens

    def summary(self):
        return {"max_model_calls": self.max_calls, "model_calls": len(self.records),
                "max_total_tokens": self.max_tokens, "accounted_total_tokens": self.spent_tokens,
                "actual_total_tokens": sum(r["usage"].get("total_tokens", 0) or 0 for r in self.records),
                "missing_usage_calls": sum(not isinstance(r["usage"].get("total_tokens"), int) for r in self.records),
                "max_elapsed_seconds": self.max_seconds, "denied_reason": self.denied,
                "records": self.records,
                "token_limit_note": "Conservative byte-based admission estimate plus actual-usage stop; not an exact tokenizer-based hard total-token guarantee.",
                "deadline_note": "Model calls use remaining HTTP timeout. In-flight local embedding/database work is checked between nodes, not cancelled."}

    def close(self):
        if self.owned:
            self.client.close()
