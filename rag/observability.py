"""Observability bootstrap for llm_code.

Goals
- Fully airgapped: exports only to an internal OTLP endpoint (Collector) via env vars.
- Config-driven: respect OTEL_* env vars; no hard-coded endpoints.
- Low-risk defaults: do NOT capture prompts/responses; do not add custom payload attributes.
- Idempotent: safe to call multiple times.

Notes
- This file intentionally avoids importing application modules to prevent import cycles.
- FastAPI/httpx/etc instrumentation should be called by the app entrypoints.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from opentelemetry import _logs, metrics, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor


# Module-level guard for idempotent init
_INITIALIZED = False


def _truthy_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _otel_disabled() -> bool:
    # Standard env var is OTEL_SDK_DISABLED=true
    return _truthy_env("OTEL_SDK_DISABLED", default=False)


def _enable_python_log_correlation() -> bool:
    # When true, inject trace/span context into stdlib logging records.
    # We default to False to avoid surprising formatting changes.
    return _truthy_env("OTEL_PYTHON_LOG_CORRELATION", default=False)


def _attach_otel_logging_handler(logger_provider: LoggerProvider) -> None:
    """Attach an OpenTelemetry LoggingHandler to common loggers.

    This causes stdlib logs (including uvicorn) to be emitted as OTEL LogRecords,
    enabling export via OTEL_LOGS_EXPORTER=otlp to the Collector (and then Loki).

    Idempotent: does not add duplicate handlers.
    """

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

    for logger_name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(logger_name)
        if any(isinstance(h, LoggingHandler) for h in lg.handlers):
            continue
        lg.addHandler(handler)


def _resource(service_name: str, service_version: Optional[str] = None) -> Resource:
    attrs = {
        "service.name": service_name,
        # You can override via OTEL_RESOURCE_ATTRIBUTES; SDK merges automatically.
    }
    if service_version:
        attrs["service.version"] = service_version
    return Resource.create(attrs)


def _should_export(kind: str) -> bool:
    """Return True if OTEL_*_EXPORTER indicates we should export for this signal.

    kind: one of "traces", "metrics", "logs"

    Behavior:
    - If OTEL_<KIND>_EXPORTER is set to "none", do not export.
    - Otherwise default to exporting (common setups set OTEL_EXPORTER_OTLP_ENDPOINT).
    """
    key = f"OTEL_{kind.upper()}_EXPORTER"
    v = os.getenv(key)
    if v is None:
        # Default: enable exporters; export will be a no-op if endpoint isn't reachable.
        return True
    return v.strip().lower() != "none"


def setup_observability(
    *,
    service_name: str = "llm_code",
    service_version: Optional[str] = None,
) -> bool:
    """Initialize OpenTelemetry providers.

    Returns:
        True if initialization happened in this call, False if already initialized or disabled.
    """
    global _INITIALIZED

    if _INITIALIZED:
        return False

    if _otel_disabled():
        # Respect OTEL_SDK_DISABLED=true
        _INITIALIZED = True
        return False

    res = _resource(service_name, service_version)

    # --- Traces ---
    try:
        tracer_provider = TracerProvider(resource=res)
        if _should_export("traces"):
            span_exporter = OTLPSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
    except Exception:
        # Do not crash the app if telemetry can't be configured.
        pass

    # --- Metrics ---
    try:
        readers = []
        if _should_export("metrics"):
            metric_exporter = OTLPMetricExporter()
            readers.append(PeriodicExportingMetricReader(metric_exporter))
        meter_provider = MeterProvider(resource=res, metric_readers=readers)
        metrics.set_meter_provider(meter_provider)
    except Exception:
        pass

    # --- Logs ---
    try:
        logger_provider = LoggerProvider(resource=res)
        if _should_export("logs"):
            log_exporter = OTLPLogExporter()
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        _logs.set_logger_provider(logger_provider)

        # Route stdlib logging into OTEL logs when enabled.
        if _should_export("logs"):
            _attach_otel_logging_handler(logger_provider)

        # Optional: correlate stdlib logging with traces (adds trace_id/span_id fields).
        # This does NOT capture prompts/responses; it only enriches log records.
        if _enable_python_log_correlation():
            LoggingInstrumentor().instrument(set_logging_format=True)
    except Exception:
        # Do not crash the app if telemetry/log correlation can't be configured.
        pass

    _INITIALIZED = True
    return True

def is_observability_initialized() -> bool:
    return _INITIALIZED
