from flask import Flask, request, jsonify
import json
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# === CONFIGURACIÓN ===
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

INDEX_FILE = "vector_index.faiss"
METADATA_FILE = "metadata.json"

app = Flask(__name__)
from flask import send_from_directory

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# === FUNCIONES ===

def obtener_embedding(texto):
    response = genai.embed_content(
        model="models/embedding-001",
        content=texto,
        task_type="RETRIEVAL_QUERY"
    )
    return np.array(response["embedding"], dtype=np.float32)

def cargar_index_y_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)["metadatos"]
    return index, metadata

def buscar_contexto_para_gemini(consulta, top_k=3):
    index, metadata = cargar_index_y_metadata()
    if index is None:
        return ""
    vector_consulta = obtener_embedding(consulta)
    D, I = index.search(np.array([vector_consulta]), k=top_k)

    contexto = ""
    for idx in I[0]:
        doc = metadata[idx]
        contexto += f"Documento: {doc['documento']}\nTexto: {doc['texto']}\n\n"
    return contexto

def responder_con_gemini(pregunta, contexto):
    modelo = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
Usá el siguiente contexto para responder la pregunta del usuario.

Contexto:
{contexto}

Pregunta:
{pregunta}
"""
    respuesta = modelo.generate_content(prompt)
    return respuesta.text

# === RUTA PRINCIPAL ===
@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.json
    pregunta = data.get("pregunta", "")
    if not pregunta:
        return jsonify({"error": "Falta la pregunta"}), 400

    contexto = buscar_contexto_para_gemini(pregunta)
    respuesta = responder_con_gemini(pregunta, contexto)
    return jsonify({"respuesta": respuesta})

# === MAIN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port)
