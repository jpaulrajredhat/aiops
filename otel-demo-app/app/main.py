import logging
import random
import time
import math
import gc
from time import sleep
import psutil

from fastapi import FastAPI

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.metrics import set_meter_provider, get_meter

from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter


# Resource definition
resource = Resource(attributes={"service.name": "anomaly-detection"})

# ----- TRACING -----
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)
trace_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
span_processor = BatchSpanProcessor(trace_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# ----- METRICS -----
metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4318/v1/metrics")
reader = PeriodicExportingMetricReader(metric_exporter)
meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(meter_provider)
meter = get_meter("anomaly-metrics")

# Define metric callbacks for CPU and memory usage
def cpu_usage_callback():
    return [
        psutil.cpu_percent(interval=None)
    ]

def memory_usage_callback():
    mem = psutil.virtual_memory()
    return [mem.used]

# Create observable metrics (CPU and memory usage)
meter.create_observable_gauge("app_cpu_usage_percent", unit="%", callbacks=[cpu_usage_callback])
meter.create_observable_gauge("app_memory_usage_bytes", unit="By", callbacks=[memory_usage_callback])

# Create custom metrics for endpoint performance
cpu_intensive_counter = meter.create_counter("cpu_intensive_requests_total", description="Total CPU intensive requests")
memory_intensive_counter = meter.create_counter("memory_intensive_requests_total", description="Total memory intensive requests")
endpoint_duration_histogram = meter.create_histogram("endpoint_duration_seconds", description="Endpoint response time in seconds")

# ----- LOGGING -----
log_exporter = OTLPLogExporter(endpoint="http://otel-collector:4318/v1/logs")
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

otel_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
logging.basicConfig(level=logging.INFO, handlers=[otel_handler])
logger = logging.getLogger("anomaly-logger")

# ----- FASTAPI -----
app = FastAPI()
FastAPIInstrumentor().instrument_app(app)

@app.get("/")
def read_root():
    with tracer.start_as_current_span("root-span"):
        logger.info("Handling root request")
        sleep(random.uniform(0.1, 0.3))
        return {"message": "Hello from OpenTelemetry instrumented FastAPI with logs!"}

@app.get("/cpu-intensive")
def get_cpu_intensive():
    start_time = time.time()
    cpu_intensive_counter.add(1)
    
    with tracer.start_as_current_span("cpu-intensive-operation"):
        logger.info("Starting CPU-intensive computation")
        
        # CPU-intensive mathematical computations
        result = 0
        iterations = 1000000  # Adjust this number to control CPU load
        
        for i in range(iterations):
            # Perform various mathematical operations
            result += math.sqrt(i) * math.sin(i) * math.cos(i)
            result += math.pow(i, 0.5) * math.log(i + 1)
            result += math.factorial(i % 10)  # Factorial of small numbers
            result += math.gcd(i, 1000)  # Greatest common divisor
        
        # Additional CPU-intensive operations
        fibonacci_sequence = []
        a, b = 0, 1
        for _ in range(1000):  # Generate Fibonacci sequence
            fibonacci_sequence.append(a)
            a, b = b, a + b
        
        # Prime number calculation
        primes = []
        for num in range(2, 1000):
            is_prime = True
            for i in range(2, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
        
        duration = time.time() - start_time
        endpoint_duration_histogram.record(duration, {"endpoint": "cpu-intensive"})
        
        logger.info(f"CPU-intensive computation completed in {duration:.2f} seconds")
        
        return {
            "message": "CPU-intensive computation completed",
            "result": result,
            "fibonacci_count": len(fibonacci_sequence),
            "primes_found": len(primes),
            "duration_seconds": duration,
            "cpu_usage_percent": psutil.cpu_percent(interval=None)
        }

@app.get("/memory-intensive")
def get_memory_intensive():
    start_time = time.time()
    memory_intensive_counter.add(1)
    
    with tracer.start_as_current_span("memory-intensive-operation"):
        logger.info("Starting memory-intensive operation")
        
        # Memory-intensive data structures
        large_lists = []
        large_dicts = []
        
        # Create multiple large data structures
        for i in range(10):  # Create 10 large data structures
            # Large list with random data
            large_list = [random.random() for _ in range(100000)]
            large_lists.append(large_list)
            
            # Large dictionary with nested structures
            large_dict = {
                f"key_{j}": {
                    "nested_data": [random.randint(1, 1000) for _ in range(1000)],
                    "metadata": f"item_{j}",
                    "timestamp": time.time()
                }
                for j in range(1000)
            }
            large_dicts.append(large_dict)
        
        # Create a large matrix
        matrix_size = 1000
        matrix = [[random.random() for _ in range(matrix_size)] for _ in range(matrix_size)]
        
        # Perform some operations on the data
        total_sum = sum(sum(lst) for lst in large_lists)
        matrix_sum = sum(sum(row) for row in matrix)
        
        # Get memory usage before cleanup
        memory_before_cleanup = psutil.virtual_memory().used
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # Clean up some data to show memory fluctuation
        del large_lists[5:]  # Delete half of the lists
        del large_dicts[5:]  # Delete half of the dictionaries
        gc.collect()  # Force garbage collection
        
        # Get memory usage after cleanup
        memory_after_cleanup = psutil.virtual_memory().used
        
        duration = time.time() - start_time
        endpoint_duration_histogram.record(duration, {"endpoint": "memory-intensive"})
        
        logger.info(f"Memory-intensive operation completed in {duration:.2f} seconds")
        
        return {
            "message": "Memory-intensive operation completed",
            "data_structures_created": len(large_lists) + len(large_dicts),
            "matrix_size": f"{matrix_size}x{matrix_size}",
            "total_sum": total_sum,
            "matrix_sum": matrix_sum,
            "memory_before_cleanup_bytes": memory_before_cleanup,
            "memory_after_cleanup_bytes": memory_after_cleanup,
            "memory_freed_bytes": memory_before_cleanup - memory_after_cleanup,
            "duration_seconds": duration,
            "current_memory_usage_bytes": psutil.virtual_memory().used,
            "memory_usage_percent": psutil.virtual_memory().percent
        }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    with tracer.start_as_current_span("health-check"):
        logger.info("Health check requested")
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "system_metrics": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_bytes": memory.available,
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "disk_free_bytes": disk.free
            }
        }

@app.get("/metrics/summary")
def get_metrics_summary():
    """Endpoint to get current system metrics summary"""
    with tracer.start_as_current_span("metrics-summary"):
        logger.info("Metrics summary requested")
        
        return {
            "timestamp": time.time(),
            "cpu_usage_percent": psutil.cpu_percent(interval=None),
            "memory_usage": {
                "total_bytes": psutil.virtual_memory().total,
                "available_bytes": psutil.virtual_memory().available,
                "used_bytes": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent
            },
            "disk_usage": {
                "total_bytes": psutil.disk_usage('/').total,
                "used_bytes": psutil.disk_usage('/').used,
                "free_bytes": psutil.disk_usage('/').free,
                "percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            }
        }
