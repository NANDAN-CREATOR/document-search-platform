"""Arize Phoenix tracing setup - lightweight client only."""
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from config.settings import settings

logger = logging.getLogger(__name__)
_tracer_provider = None


def setup_tracer_provider() -> TracerProvider:
    global _tracer_provider
    if _tracer_provider:
        return _tracer_provider

    provider = TracerProvider()

    # Try Phoenix OTLP exporter first, fall back to console
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        exporter = OTLPSpanExporter(
            endpoint=f"http://{settings.phoenix_host}:{settings.phoenix_grpc_port}"
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(f"Phoenix tracing enabled at {settings.phoenix_host}:{settings.phoenix_grpc_port}")
    except Exception:
        # Fall back to console exporter if Phoenix not running
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.warning("Phoenix not available - using console span exporter")

    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    return provider


def instrument_llamaindex() -> None:
    provider = setup_tracer_provider()
    try:
        LlamaIndexInstrumentor().instrument(tracer_provider=provider)
        logger.info("LlamaIndex instrumented with tracing.")
    except Exception as e:
        logger.warning(f"LlamaIndex instrumentation failed: {e}")


def instrument_all() -> None:
    instrument_llamaindex()
    logger.info("Tracing instrumentation complete.")
