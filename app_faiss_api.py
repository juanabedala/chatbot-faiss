import os
from flask import Flask, request, jsonify, send_from_directory

print("🔥 Starting Flask app...")

app = Flask(__name__)

@app.route("/")
def index():
    print("📥 Serving index.html")
    return send_from_directory("static", "index.html")

@app.route("/preguntar", methods=["POST"])
def preguntar():
    print("📨 Recibida pregunta")
    data = request.json
    pregunta = data.get("pregunta", "")
    print(f"Pregunta: {pregunta}")

    # Aquí ponés tu lógica para buscar en FAISS y llamar a Gemini
    # Por ahora devolvemos un dummy para probar el flujo
    respuesta = f"Respuesta simulada a: {pregunta}"

    print(f"📤 Respuesta: {respuesta}")
    return jsonify({"respuesta": respuesta})


# === MAIN ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"🚀 App corriendo en puerto {port}")
    app.run(host="0.0.0.0", port=port)
