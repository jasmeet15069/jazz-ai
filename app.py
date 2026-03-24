from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
import torch
import os
import traceback
import uuid

app = Flask(__name__)

# -----------------------------
# CPU Optimization
# -----------------------------
cpu_count = os.cpu_count()
threads = max(1, cpu_count // 2)

os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

torch.set_num_threads(threads)

print(f"CPU cores: {cpu_count}, using {threads} threads")

# -----------------------------
# Vector Database
# -----------------------------
print("Loading vector database...")

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="chat_memory")

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Vector database ready")

# -----------------------------
# Load LLM
# -----------------------------
MODEL_ID = "huihui-ai/Qwen3-1.7B-abliterated"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.pad_token_id = tokenizer.eos_token_id

print("Model loaded successfully")

# -----------------------------
# Retrieve context
# -----------------------------
def retrieve_context(query):

    try:

        query_embedding = embed_model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        docs = results.get("documents", [[]])[0]

        context = "\n".join(docs)

        return context

    except:
        return ""


# -----------------------------
# Save conversation
# -----------------------------
def save_memory(prompt, response):

    text = f"User: {prompt}\nAssistant: {response}"

    embedding = embed_model.encode(text).tolist()

    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(uuid.uuid4())]
    )


# -----------------------------
# Generate response
# -----------------------------
def generate_response(user_prompt, max_new_tokens=150):

    context = retrieve_context(user_prompt)

    prompt = f"""
You are Jazz, a confident, witty, intelligent AI assistant.

You speak naturally like a human and give clear structured answers.

Conversation memory:
{context}

User: {user_prompt}
Jazz:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.inference_mode():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Jazz:" in generated_text:
        generated_text = generated_text.split("Jazz:")[-1]

    generated_text = generated_text.replace("<|imagination|>", "").strip()

    return generated_text


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():

    try:

        data = request.get_json(force=True)
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "Empty prompt"}), 400

        response = generate_response(prompt)

        save_memory(prompt, response)

        return jsonify({
            "prompt": prompt,
            "response": response
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000
    )
