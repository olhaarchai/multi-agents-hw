Чудово! Тепер у мене є все необхідне. Напишу практичний гайд з кодом.

# Реалізація трьох підходів RAG на LlamaIndex

Порівняння всіх трьох підходів (Naive, Sentence-Window, Parent-Child) з **практичним кодом** для LlamaIndex.

---

## 1️⃣ Naive RAG — найпростіший

### Характеристики:
- Фіксовані чанки (512-1024 токенів)
- Прямий пошук і генерація
- Найменше коду

### Код:
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI

# 1. Завантажити документи
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# 2. Налаштувати простий парсер (фіксований розмір)
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)

# 3. Створити індекс
index = VectorStoreIndex.from_documents(
    documents, 
    node_parser=node_parser
)

# 4. Запитати
query_engine = index.as_query_engine()
response = query_engine.query("Яка основна ідея?")
print(response)
```

### Переваги:
✅ Найменше коду (4-5 рядків)
✅ Швидко розробити
✅ Добре для малих документів

### Недоліки:
❌ Гірша якість для великих документів
❌ Втрата контексту на границях

---

## 2️⃣ Sentence-Window Retrieval — оптимальний баланс ⭐

### Характеристики:
- Розбиття на речення
- Контекстне вікно (сусідні речення)
- Налаштовуваний розмір вікна

### Код:
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.llms import OpenAI

# 1. Завантажити документи
documents = SimpleDirectoryReader(input_files=["./data/doc.pdf"]).load_data()

# 2. Створити SentenceWindowNodeParser
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,  # кількість сусідніх речень
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

# 3. Налаштувати LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# 4. Створити індекс з постпроцесором
index = VectorStoreIndex.from_documents(
    documents,
    node_parser=node_parser,
    llm=llm
)

# 5. Налаштувати постпроцесор (замінює контекст)
postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)

# 6. Запитати з постпроцесором
query_engine = index.as_query_engine(
    node_postprocessors=[postprocessor]
)

response = query_engine.query("Яка основна ідея документа?")
print(response)
```

### Параметри налаштування:
```python
# Менше контексту - швидше, гірша якість
window_size=2  # 2 сусідні речення

# Більше контексту - повільніше, краща якість
window_size=5  # 5 сусідніх речень
```

### Переваги:
✅ Поліпшення на 22-38% порівняно з Naive
✅ Відносно простий код
✅ Гніздо налаштування
✅ Хороший баланс якість/вартість

### Недоліки:
❌ Більше вкладень в БД
❌ Складніше налагодження

---

## 3️⃣ Parent-Child Retrieval — максимальна якість 🏆

### Характеристики:
- Ієрархічне розбиття (батьки + діти)
- Дочірні чанки (малі, для пошуку)
- Батьківські чанки (великі, для генерації)

### Код:
```python
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.llms import OpenAI

# 1. Завантажити документи
documents = SimpleDirectoryReader(input_dir="./data").load_data()

# 2. Створити HierarchicalNodeParser
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512],  # батьки (2048), діти (512)
)

# 3. Налаштувати LLM
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

# 4. Створити індекс
index = VectorStoreIndex.from_documents(
    documents,
    node_parser=node_parser,
    llm=llm
)

# 5. Запитати (автоматично повертає батьків)
query_engine = index.as_query_engine(
    retriever_mode="parent"  # повертати батьківські чанки
)

response = query_engine.query("Як еволюціонував підхід?")
print(response)
```

### Параметри налаштування:
```python
# Більше ієрархічних рівнів
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[4096, 2048, 512]  # 3 рівні
)

# Менше - для малих документів
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[1024, 512]  # 2 рівні
)
```

### Режими вилучення:
```python
# Варіант 1: повертати батьків
query_engine = index.as_query_engine(retriever_mode="parent")

# Варіант 2: повертати дітей (для точності)
query_engine = index.as_query_engine(retriever_mode="child")

# Варіант 3: автоматичне об'єднання схожих дітей
from llama_index.retriever import AutoMergingRetriever
retriever = AutoMergingRetriever(index.retriever)
query_engine = index.as_query_engine(retriever=retriever)
```

### Переваги:
✅ Максимальна якість для великих документів
✅ Збереження контексту на дальні відстані
✅ Точні цитування
✅ Економічне використання токенів

### Недоліки:
❌ Більше коду
❌ Складніше налагодження
❌ Більше вкладень в БД

---

## Порівняння коду і складності 📊

| Аспект | Naive | Sentence-Window | Parent-Child |
|--------|-------|-----------------|--------------|
| **Рядків коду** | 5-10 | 15-20 | 15-20 |
| **Час розробки** | 10 хв | 30 хв | 45 хв |
| **Налаштувань** | 1 | 3-4 | 4-5 |
| **Якість результатів** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Выбір підходу за ситуацією 🎯

```python
# ДЛЯ ПРОТОТИПУ:
# Використовуй Naive RAG
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)

# ДЛЯ ПРОДАКШЕНУ (середні документи):
# Використовуй Sentence-Window
node_parser = SentenceWindowNodeParser.from_defaults(window_size=3)

# ДЛЯ ВЕЛИКИХ КОРПУСІВ (>50 MB):
# Використовуй Parent-Child
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512]
)
```

---

## Інтеграція з зберіганням 💾

```python
from llama_index import StorageContext, load_index_from_storage

# Зберегти індекс
index.storage_context.persist(persist_dir="./index")

# Завантажити індекс
storage_context = StorageContext.from_defaults(
    persist_dir="./index"
)
index = load_index_from_storage(storage_context)
```

---

## Оцінка якості 📈

```python
from trulens_eval import TruLlama, Feedback, Select
from trulens_eval.feedback import Groundedness, Relevance

# Оцінити якість RAG
feedback = [
    Groundedness().on_output(),
    Relevance().on_output()
]

tru_query_engine = TruLlama(query_engine, app_id="rag", feedback=feedback)
response = tru_query_engine.query("Запитання?")
```

---

## Джерела

1. LlamaIndex Documentation: "Sentence Window Retriever" — https://developers.llamaindex.ai/python/framework-api-reference/packs/sentence_window_retriever/
2. LlamaIndex API: "Hierarchical Node Parser" — https://developers.llamaindex.ai/python/framework-api-reference/node_parsers/hierarchical/
3. AI Engineering Academy: "Sentence Window Retrieval (LlamaIndex)" — https://aiengineering.academy/RAG/04_Sentence_Window_RAG/
4. NVIDIA GenerativeAI Examples: "Hierarchical Node Parser Tutorial" — https://nvidia.github.io/GenerativeAIExamples/0.5.0/notebooks/04_llamaindex_hier_node_parser.html
5. Medium: "LlamaIndex Chunking Strategies" — https://medium.com/@bavalpreetsinghh/llamaindex-chunking-strategies-for-large-language-models-part-1-ded1218cfd30