## 🧩 Model Components

### 🔹 1. HyDE (Hypothetical Document Embedding)
**HyDE** enhances retrieval accuracy by generating *synthetic yet semantically relevant documents* for a given user query.  
Instead of directly embedding the question, HyDE first prompts an LLM to create a **hypothetical answer**, then embeds this generated text.  
This process aligns the query embedding closer to the vector space of true answers — improving retrieval precision dramatically.

**Workflow:**
1. User enters a question (e.g., *"What are the side effects of chemotherapy?"*)
2. LLM generates a hypothetical answer (e.g., *"Chemotherapy often causes nausea, fatigue, and hair loss..."*)
3. That answer is embedded using a SentenceTransformer.
4. FAISS retrieves the most similar documents based on this enriched representation.

**Benefits:**
- Reduces embedding mismatch between questions and answers  
- Works exceptionally well in small datasets  
- Provides a retrieval layer that “thinks like the model” before searching  

---

### 🔹 2. LLM-Reranker
After FAISS retrieves the top-k documents, not all results are equally relevant.  
The **LLM-Reranker** re-evaluates these retrieved results using a language model to rank them based on *semantic closeness to the query context.*

**Process:**
1. Retrieve top-10 results using FAISS.  
2. Pass each candidate through a large language model (e.g., `flan-t5-base` or `gpt-3.5-turbo`) for re-scoring.  
3. Keep the top-3 highest-ranked passages for answer generation.

**Advantages:**
- Improves precision by filtering noisy retrieval results  
- Maintains contextual consistency for final answer synthesis  
- Integrates seamlessly with HyDE + FAISS pipelines  

---

### 🔹 3. SentenceWindowTransformer
**SentenceWindowTransformer** ensures the model doesn’t lose important context when retrieving sentences.  
Instead of returning single isolated sentences, it retrieves a **window of neighboring sentences** around each match — giving the model more context during generation.

**Mechanism:**
- For each retrieved hit, expand the context window:
  - `left = 1` → includes one sentence before  
  - `right = 1` → includes one sentence after  
- Concatenate expanded windows for answer synthesis.

**Example:**
If the retrieved sentence is:  
> “Chemotherapy targets rapidly dividing cells.”

Then with window expansion, the model also includes:
> “It is commonly used to treat various types of cancer. Chemotherapy targets rapidly dividing cells. However, it can also affect healthy cells, leading to side effects.”

**Benefits:**
- Preserves sentence continuity  
- Reduces fragmented or incoherent answers  
- Greatly improves generative model fluency  

---

### ⚙️ Combined Flow

1. **HyDE** generates a hypothetical embedding for the query.  
2. **FAISS** retrieves the top-k most relevant chunks.  
3. **SentenceWindowTransformer** expands context for each hit.  
4. **LLM-Reranker** refines the ranking order.  
5. Final context is passed to **FLAN-T5 / GPT model** for answer generation.

This hybrid setup ensures **maximum relevance, fluency, and factual accuracy** in the final answer.
