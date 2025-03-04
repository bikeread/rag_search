"""
Prometheus metrics collection for system monitoring.
"""

from prometheus_client import Counter, Histogram, start_http_server
from config.config import PROMETHEUS_PORT
import time

# Define metrics
REQUESTS_TOTAL = Counter(
    'qa_requests_total',
    'Total number of Q&A requests processed'
)

RESPONSE_TIME = Histogram(
    'qa_response_time_seconds',
    'Time taken to generate answers',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
)

DOCUMENTS_ADDED = Counter(
    'documents_added_total',
    'Total number of documents added to the knowledge base'
)

RETRIEVAL_TIME = Histogram(
    'document_retrieval_time_seconds',
    'Time taken to retrieve relevant documents',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, float('inf')]
)

class MetricsCollector:
    """Collects and exposes system metrics."""
    
    def __init__(self, port: int = PROMETHEUS_PORT):
        """Initialize metrics collector.
        
        Args:
            port: Port to expose metrics on
        """
        self.port = port
        start_http_server(port)
    
    def record_request(self):
        """Record a new Q&A request."""
        REQUESTS_TOTAL.inc()
    
    def record_response_time(self, start_time: float):
        """Record time taken to generate an answer.
        
        Args:
            start_time: Start time of the request
        """
        RESPONSE_TIME.observe(time.time() - start_time)
    
    def record_documents_added(self, count: int):
        """Record number of documents added.
        
        Args:
            count: Number of documents added
        """
        DOCUMENTS_ADDED.inc(count)
    
    def record_retrieval_time(self, start_time: float):
        """Record time taken for document retrieval.
        
        Args:
            start_time: Start time of retrieval
        """
        RETRIEVAL_TIME.observe(time.time() - start_time) 