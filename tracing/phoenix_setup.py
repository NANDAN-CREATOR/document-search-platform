import logging
import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.crewai import CrewAIInstrumentor
from config.settings import settings

logger = logging.getLogger(__name__)
_tracer_provider = None

def init_phoenix():
    try:
        session = px.launch_app()
        logger.info(f"Phoenix running at: {session.url}")
    except Exception as e:
        logger.warning(f"Phoenix already running: {e}")

def setup_tracer_provider() -> TracerProvider:
    global _tracer_provider
    if _tracer_provider:
        return _tracer_provider
    exporter = OTLPSpanExporter(endpoint=f"http://{settings.phoenix_host}:{settings.phoenix_grpc_port}")
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider
    logger.info("Tracer provider configured.")
    return provider

def instrument_llamaindex():
    provider = setup_tracer_provider()
    LlamaIndexInstrumentor().instrument(tracer_provider=provider)
    logger.info("LlamaIndex instrumented.")

def instrument_crewai():
    provider = setup_tracer_provider()
    CrewAIInstrumentor().instrument(tracer_provider=provider)
    logger.info("CrewAI instrumented.")

def instrument_all():
    setup_tracer_provider()
    instrument_llamaindex()
    instrument_crewai()
    logger.info("All frameworks instrumented.")
