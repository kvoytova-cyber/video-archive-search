from flask import Flask, request, render_template_string
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

app = Flask(__name__)

print("Загружаю индекс...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("Готово!")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Видеоархив Царского Села</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #2c3e50; }
        input[type=text] { width: 70%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; font-size: 16px; background: #3498db; color: white; border: none; cursor: pointer; }
        .result { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .filename { color: #3498db; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🎬 Поиск по видеоархиву</h1>
    <form method="POST">
        <input type="text" name="query" placeholder="Например: Янтарная комната" value="{{ query }}">
        <button type="submit">Найти</button>
    </form>
    {% if results %}
    <h2>Результаты:</h2>
    {% for doc, score in results %}
    <div class="result">
        <div class="filename">📹 {{ doc.metadata.filename }}</div>
        <p>{{ doc.page_content }}</p>
    </div>
    {% endfor %}
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    query = ""
    results = []
    if request.method == "POST":
        query = request.form.get("query", "")
        if query:
            results = vector_store.similarity_search_with_score(query, k=5)
    return render_template_string(HTML, query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)