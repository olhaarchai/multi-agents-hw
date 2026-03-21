Чудово! Тепер у мене достатньо інформації. Напишу детальний практичний гайд.

# LlamaIndex RAG: Налаштування, Оцінка & Інтеграція

Докладний гайд з кодом для оптимізації, тестування та розгортування RAG систем.

---

## Частина 1️⃣: Оцінка якості RAG (RAG Triad)

### RAG Triad: 3 ключові метрики

```
┌─────────────────────────────────────┐
│       HALLUCINATION CHECK           │
├─────────────────────────────────────┤
│ 1. CONTEXT RELEVANCE                │
│    (Чи релевантна отримана інформація?)
│                                     │
│ 2. GROUNDEDNESS                     │
│    (Чи основана відповідь на контексті?)
│                                     │
│ 3. ANSWER RELEVANCE                 │
│    (Чи відповідь відповідає запиту?)
└─────────────────────────────────────┘
```

### Встановлення TruLens:
```bash
pip install trulens_eval llama_index openai
```

### Повна реалізація оцінки:

```python
import os
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from trulens_eval import TruLlama, Feedback, Tru
from trulens_eval.feedback import (
    Groundedness, 
    Relevance, 
    ContextRelevance
)
import numpy as np

# 1. НАЛАШТУВАТИ DOCUMENTS & INDEX
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 2. ВИЗНАЧИТИ FEEDBACK ФУНКЦІЇ
groundedness = Groundedness(
    define_by_llm=True  # використовувати LLM для оцінки
)
context_relevance = ContextRelevance()
relevance = Relevance()

# 3. СТВОРИТИ FEEDBACK СПИСОК
feedback = [
    groundedness.on_output().tag_index("groundedness"),
    context_relevance.on_document_source().tag_index("context_relevance"),
    relevance.on_output().tag_index("answer_relevance")
]

# 4. ОБГОРНУТИ QUERY ENGINE
tru_query_engine = TruLlama(
    query_engine,
    app_id="my_rag_app",
    feedback=feedback
)

# 5. ЗАПИТАТИ З ОЦІНКОЮ
test_queries = [
    "Яка основна ідея документа?",
    "Хто автор?",
    "Як це практично застосовується?"
]

results = []
for query in test_queries:
    response = tru_query_engine.query(query)
    results.append({
        "query": query,
        "response": str(response),
    })
    print(f"✅ Query: {query}")
    print(f"   Response: {response}\n")

# 6. ОТРИМАТИ МЕТРИКИ
tru = Tru()
records, feedback_scores = tru.get_records_and_feedback(app_ids=["my_rag_app"])

# Вивести результати
print("\n" + "="*60)
print("📊 RAG EVALUATION RESULTS")
print("="*60)
for feedback_score in feedback_scores:
    print(f"\n{feedback_score.feedback_definition.name}:")
    print(f"  Average Score: {np.mean([f.result for f in feedback_scores]):.2f}")
    for score in feedback_score:
        print(f"  - {score.result:.2f}")
```

### Інтерпретація результатів:

| Метрика | Оцінка | Значення |
|---------|--------|----------|
| **Context Relevance** | 0.8-1.0 | ✅ Рівень доброго пошуку |
| | 0.5-0.8 | ⚠️ Потребує поліпшення |
| | < 0.5 | ❌ Поганий пошук |
| **Groundedness** | 0.8-1.0 | ✅ Відповідь на основі контексту |
| | 0.5-0.8 | ⚠️ Деякі галюцинації |
| | < 0.5 | ❌ Багато галюцинацій |
| **Answer Relevance** | 0.8-1.0 | ✅ Добре відповідає запиту |
| | 0.5-0.8 | ⚠️ Частково релевантна |
| | < 0.5 | ❌ Не релевантна |

---

## Частина 2️⃣: Налаштування гіперпараметрів

### Параметри для оптимізації:

```
┌──────────────────────────┐
│  CHUNK_SIZE              │  256 → 512 → 1024 → 2048
├──────────────────────────┤
│  WINDOW_SIZE             │  1 → 3 → 5 → 10 (для Sentence-Window)
├──────────────────────────┤
│  TOP_K                   │  1 → 3 → 5 → 10 (кількість чанків)
├──────────────────────────┤
│  SIMILARITY_THRESHOLD    │  0.5 → 0.6 → 0.7 → 0.8
└──────────────────────────┘
```

### Автоматичне налаштування (Grid Search):

```python
from llama_index.param_tuner import ParamTuner
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import numpy as np

# 1. ВИЗНАЧИТИ RAG ФУНКЦІЮ
def rag_pipeline(chunk_size: int, top_k: int, documents: list):
    """Функція для налаштування"""
    from llama_index.node_parser import SimpleNodeParser
    
    # Розбити документи
    parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    nodes = parser.get_nodes_from_documents(documents)
    
    # Створити індекс
    index = VectorStoreIndex(nodes)
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    
    # Метрика оцінки (1 = найкраще)
    test_query = "Основна ідея?"
    response = query_engine.query(test_query)
    
    # Повернути результати
    return {
        "correctness_score": np.random.random(),  # ваша метрика
        "latency": 0.5,
        "token_cost": chunk_size * top_k
    }

# 2. ДЕФІНІЮВАТИ ПАРАМЕТРИ
documents = SimpleDirectoryReader("./data").load_data()

param_dict = {
    "chunk_size": [256, 512, 1024],
    "top_k": [2, 3, 5]
}

# 3. ЗАПУСТИТИ TUNER
tuner = ParamTuner()

results = tuner.tune(
    rag_pipeline,
    param_dict=param_dict,
    documents=documents,
    show_progress=True
)

# 4. ОТРИМАТИ НАЙКРАЩІ ПАРАМЕТРИ
print("\n" + "="*60)
print("🎯 BEST PARAMETERS")
print("="*60)
best_result = tuner.get_best_result()
print(f"Best chunk_size: {best_result['chunk_size']}")
print(f"Best top_k: {best_result['top_k']}")
print(f"Best score: {best_result['correctness_score']:.3f}")

# 5. ПЕРЕГЛЯНУТИ ВСІ РЕЗУЛЬТАТИ
print("\n📈 All Results:")
for params, metrics in tuner.results.items():
    print(f"  {params}: {metrics['correctness_score']:.3f}")
```

### Рекомендовані параметри по сценаріям:

```python
# Сценарій 1: Малі документи (< 10 сторінок)
config_small = {
    "chunk_size": 256,
    "window_size": 1,  # для Sentence-Window
    "top_k": 2,
    "similarity_threshold": 0.7
}

# Сценарій 2: Середні документи (10-100 сторінок)
config_medium = {
    "chunk_size": 512,
    "window_size": 3,
    "top_k": 3,
    "similarity_threshold": 0.6
}

# Сценарій 3: Великі документи (> 100 сторінок)
config_large = {
    "chunk_size": 1024,
    "window_size": 5,
    "top_k": 5,
    "similarity_threshold": 0.5,
    "use_hierarchical": True  # Parent-Child
}

# Застосувати конфіг
def apply_config(config, documents):
    from llama_index.node_parser import SimpleNodeParser
    parser = SimpleNodeParser.from_defaults(
        chunk_size=config["chunk_size"]
    )
    index = VectorStoreIndex.from_documents(documents, node_parser=parser)
    return index.as_query_engine(
        similarity_top_k=config["top_k"],
        node_postprocessors=[
            # тут інші налаштування
        ]
    )
```

---

## Частина 3️⃣: Інтеграція з векторними БД

### Варіант 1: Pinecone (Хмара)

```bash
pip install pinecone-client
```

```python
import os
from pinecone import Pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex, StorageContext, SimpleDirectoryReader

# 1. НАЛАШТУВАТИ PINECONE
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "rag-documents"

pc = Pinecone(api_key=pinecone_api_key)

# Створити індекс (якщо не існує)
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,  # OpenAI embedding size
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )

# 2. СТВОРИТИ VECTOR STORE
pinecone_index = pc.Index(pinecone_index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# 3. ЗАВАНТАЖИТИ ДОКУМЕНТИ
documents = SimpleDirectoryReader("./data").load_data()

# 4. ІНГЕСТУВАННЯ В PINECONE
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 5. ЗАПИТАТИ
query_engine = index.as_query_engine()
response = query_engine.query("Яка основна ідея?")
print(response)
```

### Варіант 2: Weaviate (Self-Hosted)

```bash
pip install weaviate-client
```

```python
import weaviate
from llama_index.vector_stores import WeaviateVectorStore
from llama_index import VectorStoreIndex, StorageContext, SimpleDirectoryReader

# 1. ПІДКЛЮЧИТИСЯ ДО WEAVIATE
client = weaviate.Client("http://localhost:8080")

# 2. СТВОРИТИ VECTOR STORE
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    class_prefix="LlamaIndex"
)

# 3. ЗАВАНТАЖИТИ ДОКУМЕНТИ
documents = SimpleDirectoryReader("./data").load_data()

# 4. ІНГЕСТУВАННЯ
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 5. ЗАПИТАТИ
query_engine = index.as_query_engine()
response = query_engine.query("Яка основна ідея?")
print(response)
```

### Порівняння БД:

| Критерій | Pinecone | Weaviate | ChromaDB |
|----------|----------|----------|----------|
| **Хмара** | ☁️ Так | ❌ Ні | ❌ Ні |
| **Масштабованість** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Вартість** | 💰💰 | Вільна | Вільна |
| **Легкість** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Частина 4️⃣: Повний Production Pipeline

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.vector_stores import PineconeVectorStore
from trulens_eval import TruLlama, Feedback
from trulens_eval.feedback import Groundedness, Relevance
import os

class ProductionRAGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.index = None
        self.query_engine = None
    
    def build_pipeline(self):
        """Побудувати повний pipeline"""
        # 1. Завантажити документи
        documents = SimpleDirectoryReader(self.config["data_dir"]).load_data()
        
        # 2. Налаштувати парсер
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.config["window_size"]
        )
        
        # 3. Налаштувати vector store (Pinecone)
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        pinecone_index = pc.Index(self.config["pinecone_index"])
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        
        # 4. Створити індекс
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            node_parser=node_parser
        )
        
        # 5. Створити query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.config["top_k"]
        )
        
        print("✅ RAG Pipeline built successfully!")
    
    def evaluate(self, test_queries: list):
        """Оцінити якість"""
        groundedness = Groundedness(define_by_llm=True)
        relevance = Relevance()
        
        feedback = [
            groundedness.on_output(),
            relevance.on_output()
        ]
        
        tru_query_engine = TruLlama(
            self.query_engine,
            app_id="production_rag",
            feedback=feedback
        )
        
        for query in test_queries:
            response = tru_query_engine.query(query)
            print(f"Q: {query}\nA: {response}\n")
    
    def query(self, question: str):
        """Запитати систему"""
        return self.query_engine.query(question)

# ВИКОРИСТАННЯ:
config = {
    "data_dir": "./data",
    "window_size": 3,
    "top_k": 5,
    "pinecone_index": "rag-production"
}

pipeline = ProductionRAGPipeline(config)
pipeline.build_pipeline()

# Оцінити
test_queries = ["Яка основна ідея?", "Як це застосовується?"]
pipeline.evaluate(test_queries)

# Запитати
response = pipeline.query("Деталізуйте підхід")
print(response)
```

---

## Чек-лист для Production 🚀

```
Налаштування:
☐ Встановлено всі залежності
☐ API ключі налаштовано
☐ Документи завантажено й очищено
☐ Векторна БД обрана й налаштована

Оцінка:
☐ Context Relevance > 0.7
☐ Groundedness > 0.75
☐ Answer Relevance > 0.8
☐ Латентність < 500ms
☐ Вартість токенів оптимізована

Деплойменту:
☐ Індекс збережено
☐ Конфігурація версіонована
☐ Лог-системи налаштовано
☐ Моніторинг увімкнено
☐ Резервні копії готові
```

---

## Джерела

1. AnalyticsVidhya: "Evaluate RAG Pipelines with LlamaIndex and TRULens" — https://www.analyticsvidhya.com/blog/2024/06/rag-pipelines-with-llamaindex-and-trulens/
2. TruLens: "RAG Triad" — https://www.trulens.org/getting_started/core_concepts/rag_triad/
3. Pinecone Docs: "LlamaIndex Integration" — https://docs.pinecone.io/integrations/llamaindex
4. C# Corner: "Vector Databases (Pinecone & Weaviate)" — https://www.c-sharpcorner.com/article/how-to-implement-vector-databases-like-pinecone-or-weaviate-in-ai-applications/
5. LlamaIndex Blog: "Evaluating Ideal Chunk Size" — https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5