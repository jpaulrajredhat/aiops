from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import faiss
from langchain.schema import Document


from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

import os


os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"

import numpy as np

# Load local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize LLM model and tokenizer
model_name = "gpt2"  # Replace with a more suitable model if needed
llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token explicitly
llm_tokenizer.pad_token = llm_tokenizer.eos_token

llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)


vector_store = None  # Placeholder for lazy initialization

def get_vector_store():
   
    global vector_store
    if vector_store is None:

        documents = [
            Document(page_content="high_cpu_usage", metadata={"anomaly": "CPU utilization exceeded 95% in {pod_name} for 10 minutes.", "resolution": "1. Check CPU usage using 'oc adm top pods -n <namespace>'.\n2. Inspect current CPU limits with 'oc get deployment <deployment-name> -n <namespace> -o yaml | grep -A5 resources'.\n3. Edit deployment to increase CPU limits using 'oc edit deployment <deployment-name> -n <namespace>' and update:\n   resources:\n     requests:\n       cpu: '500m'\n     limits:\n       cpu: '1000m'\n4. Set up Horizontal Pod Autoscaler with 'oc autoscale deployment <deployment-name> --cpu-percent=80 --min=2 --max=10 -n <namespace>'.\n5. Verify HPA with 'oc get hpa -n <namespace>'.\n6. Monitor scaling activity using 'oc get pods -n <namespace> --watch'."}),
            Document(page_content="memory_leak", metadata={"anomaly": "Memory usage is continuously increasing, causing OOM kills.", "resolution": "RestRestart the pod to clear memory usage. Review memory allocations in the deployment YAML and increase if necessary. Use tools like 'heapdump' to detect potential memory leaks in the application code."}),
            Document(page_content="network_latency", metadata={"anomaly": "Intermittent connectivity issues observed between services.", "resolution": "Verify network policies and firewall rules in OpenShift. Investigate DNS resolution delays using tools like 'nslookup' and ensure services are properly registered in the service mesh."}),
            Document(page_content="disk_pressure", metadata={"anomaly": "Free up disk space by clearing logs or unnecessary files. Monitor disk usage metrics and consider increasing storage allocation in the PersistentVolumeClaim (PVC). Optimize I/O-intensive applications to minimize disk write frequency."}),
        ]
        # Embed the documents
        doc_texts = [doc.page_content for doc in documents]
        embeddings = embedding_model.encode(doc_texts, convert_to_tensor=False)

        # Initialize FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        # Initialize FAISS vector store using LangChain
        docstore = InMemoryDocstore(dict(enumerate(documents)))
        vector_store = FAISS(
            embedding_function=lambda text: embedding_model.encode([text])[0],
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: i for i in range(len(documents))}
        )
    return vector_store


def analyze_anomaly_with_llm(anomaly_data):
   
    vector_store = get_vector_store()

    if not vector_store:
        raise ValueError("Vector store is not initialized correctly.")

    query = anomaly_data["anomaly_type"]
    result = vector_store.similarity_search(query, k=1)
    
    if result:
        relevant_doc = result[0]
        resolution = relevant_doc.metadata.get("resolution", "No resolution found.")
        related_issue = relevant_doc.page_content
    else:

        resolution = "No relevant resolution found."
        related_issue = "No relevant document found."
    
    print(resolution)
    try:
        prompt = (
            f"Anomaly detected in application '{anomaly_data['app_name']}' on pod '{anomaly_data['pod_name']}' in cluster '{anomaly_data['cluster_name']}'. "
            f"Details: {anomaly_data['description']}. "
            f"Related issue: {related_issue}. "
            f"Suggested resolution: {resolution}. "
            "Also, recommend scaling actions if necessary."
        )
        response = llm_pipeline(prompt, max_length=512, num_return_sequences=1,truncation=True)    # Mock LLM pipeline for testing
        # generated_text = response[0]["generated_text"] if isinstance(response, list) else str(response)
        generated_text = response[0]["generated_text"]
    except Exception as e:
        print(f"Error in LLM pipeline: {e}")
        generated_text = "Failed to generate a response."
    print(f"{generated_text} Additionally, consider scaling resources if usage remains high.")
    return f"{generated_text} Additionally, consider scaling resources if usage remains high."
