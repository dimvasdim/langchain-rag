"""
Cost Optimization Patterns
Reducing LLM costs in production
"""

import hashlib
import time
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from config import get_embeddings, get_llm


def count_tokens(response) -> tuple[int, int]:
    """Real (input, output) token counts reported by the model."""
    usage = response.usage_metadata or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# === Model Routing ===


class ModelRouter:
    """Route queries to appropriate model based on complexity."""

    # Hypothetical cloud prices per 1k input tokens, to show what routing would
    # save with a paid API. Local Ollama models are free; latency is the real cost.
    CHEAP_COST_PER_1K = 0.00015
    EXPENSIVE_COST_PER_1K = 0.0025

    def __init__(self):
        self.cheap_model = get_llm(temperature=0)
        self.expensive_model = get_llm(temperature=0, large=True)
        # Classify with the cheap model, or routing would cost more than it saves
        self.classifier = self.cheap_model

    def classify_complexity(self, query: str) -> str:
        """Classify query complexity."""

        prompt = ChatPromptTemplate.from_template(
            """
Classify this query's complexity as 'simple' or 'complex'.

Simple: Basic facts, short answers, simple calculations
Complex: Analysis, reasoning, creative tasks, multi-step problems

Query: {query}

Respond with only: simple or complex
"""
        )

        response = self.classifier.invoke(prompt.format(query=query))
        # Small models often answer "Simple." or "simple query" rather than the bare word
        return "simple" if "simple" in response.content.lower() else "complex"

    @traceable(name="routed_query")
    def invoke(self, query: str) -> tuple[str, str, float, float]:
        """
        Route and invoke query.
        Returns: (response, model_used, hypothetical_cost, latency_s)
        """
        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            cost_per_1k = self.CHEAP_COST_PER_1K
        else:
            model = self.expensive_model
            cost_per_1k = self.EXPENSIVE_COST_PER_1K

        start = time.perf_counter()
        response = model.invoke(query)
        latency = time.perf_counter() - start

        input_tokens, _ = count_tokens(response)
        estimated_cost = (input_tokens / 1000) * cost_per_1k
        model_name = response.response_metadata.get("model_name", complexity)

        return response.content, model_name, estimated_cost, latency


def demo_model_routing():
    """Demonstrate model routing."""

    router = ModelRouter()

    queries = [
        "What is 2 + 2?",  # Simple
        "Analyze the economic implications of AI on the job market.",  # Complex
        "What color is the sky?",  # Simple
    ]

    print("Model Routing Demo:\n")

    total_cost = 0
    for query in queries:
        result, model, cost, latency = router.invoke(query)
        total_cost += cost
        print(f"Query: {query[:50]}...")
        print(f"  Model: {model}")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Hypothetical cloud cost: ${cost:.6f}")
        print(f"  Response: {result[:50]}...")

    print(f"\nTotal hypothetical cloud cost: ${total_cost:.6f}")


# === Semantic Caching ===


class SemanticCache:
    """Cache responses with semantic similarity matching."""

    def __init__(self, similarity_threshold: float = 0.9):
        self.cache = {}
        self.threshold = similarity_threshold
        self.embedder = get_embeddings()

    def _hash_query(self, query: str) -> str:
        """Create hash of normalized query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _embed(self, query: str) -> np.ndarray:
        vector = np.array(self.embedder.embed_query(query))
        return vector / np.linalg.norm(vector)

    def get(self, query: str) -> str | None:
        """Get cached response if similar query exists."""
        query_hash = self._hash_query(query)

        # Exact match (cheap, no embedding call)
        if query_hash in self.cache:
            return self.cache[query_hash]["response"]

        # Semantic match: cosine similarity against every cached query
        if not self.cache:
            return None
        query_vector = self._embed(query)
        best = max(self.cache.values(), key=lambda e: float(query_vector @ e["vector"]))
        if float(query_vector @ best["vector"]) >= self.threshold:
            return best["response"]

        return None

    def set(self, query: str, response: str):
        """Cache a response."""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = {
            "query": query,
            "response": response,
            "vector": self._embed(query),
        }

    def stats(self) -> dict:
        return {"cached_queries": len(self.cache)}


class CachedLLM:
    """LLM wrapper with caching."""

    def __init__(self):
        self.llm = get_llm(temperature=0)
        self.cache = SemanticCache()
        self.cache_hits = 0
        self.cache_misses = 0

    @traceable(name="cached_invoke")
    def invoke(self, query: str) -> tuple[str, bool]:
        """
        Invoke with caching.
        Returns: (response, from_cache)
        """
        # Check cache
        cached = self.cache.get(query)
        if cached is not None:
            self.cache_hits += 1
            return cached, True

        # Call LLM
        self.cache_misses += 1
        response = self.llm.invoke(query)
        result = response.content

        # Cache result
        self.cache.set(query, result)

        return result, False

    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


def demo_caching():
    """Demonstrate caching."""

    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Cache hit
        "What is python?",  # Cache hit (normalized)
        "Can you tell me what Python is?",  # Cache hit (semantic), if above threshold
        "What is Rust?",
    ]

    print("\nCaching Demo:\n")

    for query in queries:
        result, from_cache = llm.invoke(query)
        source = "CACHE" if from_cache else "LLM"
        print(f"[{source}] {query} -> {result[:30]}...")

    print(f"\nStats: {llm.get_stats()}")


# === Token Budgeting ===


class TokenBudget:
    """Track and limit token usage."""

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_tokens(self, text: str) -> int:
        """Rough pre-call estimate; the real count is only known after the call."""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        """Check if request is within budget."""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        return {
            **self.usage,
            "total_tokens": self.usage["total_input"] + self.usage["total_output"],
            "avg_per_request": (
                (self.usage["total_input"] + self.usage["total_output"])
                / max(self.usage["requests"], 1)
            ),
        }


class BudgetedLLM:
    """LLM with token budgeting."""

    def __init__(self, max_tokens: int = 4000):
        self.llm = get_llm(temperature=0)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str) -> str:
        # Check budget (estimate, since we must decide before calling the model)
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )

        # Execute
        response = self.llm.invoke(query)
        result = response.content

        # Record real usage reported by the model
        self.budget.record_usage(*count_tokens(response))

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():
    """Demonstrate token budgeting."""

    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",  # Within budget
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]

    print("\nToken Budgeting Demo:\n")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"✅ {query[:40]}... -> {result[:30]}...")
        except ValueError as e:
            print(f"❌ {query[:40]}... -> {e}")

    print(f"\nUsage: {llm.get_stats()}")


if __name__ == "__main__":
    # demo_model_routing()
    demo_caching()
    # demo_token_budgeting()
