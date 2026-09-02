"""Tracing setup with graceful fallback if Phoenix not available."""
import logging

logger = logging.getLogger(__name__)


def instrument_llamaindex() -> None:
    """Instrument LlamaIndex with tracing - lazy import."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        LlamaIndexInstrumentor().instrument(tracer_provider=provider)
        logger.info("LlamaIndex tracing enabled.")
    except Exception as e:
        logger.warning(f"Tracing not available: {e} — continuing without tracing.")


def instrument_all() -> None:
    """Instrument all frameworks."""
    instrument_llamaindex()
