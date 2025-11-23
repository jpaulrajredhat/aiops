import logging
import os
import time
from typing import Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

# LangChain community adapters
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

LLM_TIMEOUT = 20  # seconds

# Global singletons (lazy)
_embedding_model = None
_vector_store: Optional[FAISS] = None
_llm_pipeline = None

# Configurable model names
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# CPU-friendly generation model: change if you have a different small model available
GEN_MODEL = os.getenv("GEN_MODEL", "distilgpt2")  # distilgpt2 is small & CPU-friendly
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "256"))

# Safety: limit torch threads in container
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))


def init_pipeline():
    """Idempotent initialization of embedding model, vector store, and LLM pipeline."""
    global _embedding_model, _vector_store, _llm_pipeline

    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if _vector_store is None:
        logger.info("Building FAISS vector store (in-memory)")
        
        docs = [
            Document(page_content="high_cpu_usage", metadata={"anomaly": "CPU utilization exceeded 95% in {pod_name} for 10 minutes.", "resolution": "1. Check CPU usage using 'oc adm top pods -n <namespace>'.\n2. Inspect current CPU limits with 'oc get deployment <deployment-name> -n <namespace> -o yaml | grep -A5 resources'.\n3. Edit deployment to increase CPU limits using 'oc edit deployment <deployment-name> -n <namespace>' and update:\n   resources:\n     requests:\n       cpu: '500m'\n     limits:\n       cpu: '1000m'\n4. Set up Horizontal Pod Autoscaler with 'oc autoscale deployment <deployment-name> --cpu-percent=80 --min=2 --max=10 -n <namespace>'.\n5. Verify HPA with 'oc get hpa -n <namespace>'.\n6. Monitor scaling activity using 'oc get pods -n <namespace> --watch'."}),
            Document(page_content="memory_leak", metadata={"anomaly": "Memory usage is continuously increasing, causing OOM kills.", "resolution": "RestRestart the pod to clear memory usage. Review memory allocations in the deployment YAML and increase if necessary. Use tools like 'heapdump' to detect potential memory leaks in the application code."}),
            Document(page_content="high_cpu_and_memory", metadata={"anomaly":"CPU utilization exceeded in {pod_name} and Memory usage is continuously increasing.","resolution": "Analyze current cpu and memory usage and Take action."}),
            Document(page_content="network_latency", metadata={"anomaly": "Intermittent connectivity issues observed between services.", "resolution": "Verify network policies and firewall rules in OpenShift. Investigate DNS resolution delays using tools like 'nslookup' and ensure services are properly registered in the service mesh."}),
            Document(page_content="disk_pressure", metadata={"anomaly": "Free up disk space by clearing logs or unnecessary files. Monitor disk usage metrics and consider increasing storage allocation in the PersistentVolumeClaim (PVC). Optimize I/O-intensive applications to minimize disk write frequency."}),
        ]
        # docs = [
        #     Document(page_content="high_cpu_usage",
        #              metadata={"anomaly": "CPU utilization exceeded 95% in {pod_name} for 10 minutes.",
        #                        "resolution": "1. Check CPU usage using 'oc adm top pods -n <namespace>'.\n2. Inspect current CPU limits with 'oc get deployment <deployment-name> -n <namespace> -o yaml | grep -A5 resources'.\n3. Edit deployment to increase CPU limits and set HPA.\n"}),
        #     Document(page_content="memory_leak",
        #              metadata={"anomaly": "Memory usage continuously increasing causing OOM kills.",
        #                        "resolution": "Restart pod; run heapdump; review memory usage in code."}),
        #     Document(page_content="network_latency",
        #              metadata={"anomaly": "Intermittent connectivity issues.", "resolution": "Check network policies and DNS."}),
        #     Document(page_content="disk_pressure",
        #              metadata={"anomaly": "Disk pressure", "resolution": "Free disk space or increase PVC size."}),
        # ]
        # Build FAISS vectorstore using LangChain-community adapter
        _vector_store = FAISS.from_documents(docs, _embedding_model)
        logger.info("FAISS vector store initialized")

    if _llm_pipeline is None:
        logger.info("Initializing LLM pipeline (cpu)")
        try:
            # try recommended small model first; if not available fall back to distilgpt2
            model_name = os.getenv("GEN_MODEL", GEN_MODEL)
            logger.info("Loading generation model: %s", model_name)
            # load tokenizer + model via model name
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name)
            _llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            logger.info("LLM pipeline ready")
        except Exception as e:
            logger.exception("Failed to initialize LLM pipeline: %s", e)
            raise

    return True


def _generate_text(prompt: str, timeout: int = 60) -> str:
    """
    Safe generation wrapper with a simple timeout check.
    Transformers pipeline does not support a hard timeout parameter, so we run generation and
    rely on CPU limits / watchdog externally. Keep the generation small to avoid hangs.
    """
    global _llm_pipeline
    if _llm_pipeline is None:
        raise RuntimeError("LLM pipeline not initialized")

    start = time.time()
    try:
        out = _llm_pipeline(prompt, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=True, top_k=50, num_return_sequences=1)
    except Exception as e:
        logger.exception("LLM generation failed: %s", e)
        return "LLM generation failed."
    elapsed = time.time() - start
    logger.info("LLM generation took %.2fs", elapsed)
    if elapsed > timeout:
        logger.warning("LLM generation exceeded timeout (%ds)", timeout)
        return "LLM generation timed out."
    try:
        # different models return different shapes; try best-effort extraction
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            # huggingface pipeline returns [{'generated_text': '...'}]
            return out[0].get("generated_text", str(out[0]))
        return str(out)
    except Exception:
        return str(out)


def get_vector_store():
    """Return initialized vector store (init on demand)."""
    if _vector_store is None:
        init_pipeline()
    return _vector_store


def analyze_anomaly_with_llm(anomaly_data: dict) -> str:
    """
    Run a similarity search + LLM reasoning and return the generated text.
    anomaly_data expected keys: app_name, pod_name, cluster_name, anomaly_type, description (optional)
    """
    logger.info("analyze_anomaly_with_llm called with %s", anomaly_data)
    # Ensure pipeline is initialized
    init_pipeline()

    # Basic input checks and safe defaults
    required = ["app_name", "pod_name", "cluster_name", "anomaly_type"]
    for r in required:
        if r not in anomaly_data:
            raise ValueError(f"Missing required field: {r}")

    details = anomaly_data.get("description", "No description provided.")
    vs = get_vector_store()
    query = anomaly_data.get("anomaly_type", "")
    try:
        hits = vs.similarity_search(query, k=1)
    except Exception as e:
        logger.exception("Vector search failed: %s", e)
        hits = []

    if hits:
        doc = hits[0]
        resolution = doc.metadata.get("resolution", "No resolution found.")
        related = doc.page_content
    else:
        resolution = "No relevant resolution found."
        related = "No relevant document found."

    prompt = (
        f"Anomaly: app={anomaly_data.get('app_name')} pod={anomaly_data.get('pod_name')} cluster={anomaly_data.get('cluster_name')}. "
        f"Anomaly type: {anomaly_data.get('anomaly_type')}. Details: {details}. "
        f"Related doc: {related}. Known resolution: {resolution}. "
        "Provide a concise technical explanation, step-by-step resolution, and recommend scaling actions if necessary."
    )

    generated = _generate_text(prompt, timeout=60)
    final = f"{generated} Additionally, consider scaling resources if usage remains high."
    logger.info("Analysis result generated")
    return final
