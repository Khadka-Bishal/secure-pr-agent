"""Observability module for the PR Agent.

Provides integration with OpenTelemetry for tracing and Prometheus for metrics.
"""

import logging
import os
import warnings
from typing import Optional

from opentelemetry import trace
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)
from prometheus_client import Counter, Histogram, start_http_server

# -----------------------------------------------------------------------------
# Prometheus Metrics
# -----------------------------------------------------------------------------

REVIEW_DURATION = Histogram(
    "pr_agent_review_duration_seconds",
    "Time taken for a PR review",
    labelnames=["repo", "runner_type"],
)

FINDINGS_COUNT = Counter(
    "pr_agent_findings_total",
    "Total number of issues found",
    labelnames=["severity", "category"],
)

TOKEN_USAGE = Counter(
    "pr_agent_llm_tokens_total",
    "Total LLM tokens consumed",
    labelnames=["model", "type"],
)

# -----------------------------------------------------------------------------
# Tracing Helper
# -----------------------------------------------------------------------------


class CleanerConsoleSpanExporter(SpanExporter):
    """Custom span exporter that only logs relevant spans to keep stdout clean."""

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            status = (
                "OK"
                if span.status.status_code.name == "UNSET"
                else str(span.status.status_code.name)
            )
            # Filter distinct spans (http requests or LLM calls)
            if "http" in span.attributes or "llm.provider" in span.attributes:
                duration_ms = (span.end_time - span.start_time) / 1e6
                print(f"[Trace] {span.name} ({status}) - {duration_ms:.2f}ms")
        return SpanExportResult.SUCCESS


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------


def setup_observability(
    service_name: str = "pr-agent", prometheus_port: int = 8000
) -> trace.Tracer:
    """Initialize Tracing and Metrics subsystems."""
    
    # Silence third-party noise
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    # Suppress specific Pydantic warnings often seen with LiteLLM/LangChain
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    try:
        start_http_server(prometheus_port)
        print(f"[Observability] Prometheus metrics exposed on port {prometheus_port}")
    except Exception as e:
        print(f"[Observability] Failed to start Prometheus server: {e}")

    resource = Resource.create({"service.name": service_name})
    tracer_provider = TracerProvider(resource=resource)

    # Use our custom exporter to avoid console spam
    tracer_provider.add_span_processor(
        BatchSpanProcessor(CleanerConsoleSpanExporter())
    )

    trace.set_tracer_provider(tracer_provider)
    RequestsInstrumentor().instrument()

    return trace.get_tracer(service_name)


def get_tracer() -> trace.Tracer:
    """Retrieve the global tracer instance."""
    return trace.get_tracer("pr-agent")
