#  BITlearn

An AI-powered, Retrieval-Augmented Generation (RAG) academic learning assistant designed to help college students instantly understand complex textbook concepts. 

Traditional textbook searching (Ctrl+F) only finds exact word matches. BITlearn uses semantic search to actually *understand* the student's question, retrieve the exact relevant textbook paragraphs, and use AI to explain the concept based on the student's learning level—while mathematically proving it didn't hallucinate.

##  Key Features

* **Semantic Textbook RAG Engine:** Upload course textbooks (PDFs). The app automatically chunks and vectorizes the text. When a user asks a question, it retrieves the top 7 most relevant paragraphs to form a factual context window.
* ** Adaptive Learning Levels:** AI explanations dynamically adjust based on user proficiency:
  * *Beginner:* High-school level, heavy on analogies.
  * *Intermediate:* Standard college-level definitions.
  * *Advanced:* Highly technical, deep-dive architecture.
* ** Multilingual Support:** Explains complex technical topics in English, Hindi, Bengali, Telugu, and Tamil, while smartly keeping core engineering terminology in English.
* ** Hallucination Checker:** Uses Cosine Similarity to compare the AI's final answer against the raw textbook text. It assigns a color-coded "Hallucination Score" to guarantee absolute academic accuracy.
* **Visual Source Verification:** Extracts and displays the exact page image from the original PDF so students can verify the primary source.
* **Premium Study Guides:** A built-in, session-state authentication system. Premium users can unlock downloadable, multi-page `.txt` study guides and practice exam questions.
* ** Contextual YouTube Scraper:** Automatically scrapes and embeds the top 2 YouTube tutorials related to the specific query and subject.

##  Tech Stack

* **Frontend:** Streamlit
* **LLM Engine:** Google Gemini 2.5-flash (`google-generativeai`)
* **Vector Database:** ChromaDB 
* **Embeddings:** `all-MiniLM-L6-v2` (via SentenceTransformers)
* **Document Processing:** PyMuPDF (`fitz`)
* **Deployment:** Streamlit Community Cloud (via Git LFS)

##  System Architecture

1. **Query Expansion:** The user's raw query (e.g., "fcfs") is passed to Gemini to expand abbreviations (e.g., "First Come First Serve") for better database matching.
2. **Vector Retrieval:** The expanded query is embedded and searched against ChromaDB. The system filters by Subject and Level, scoring chunks based on keyword frequency to find the main definition.
3. **Strict Generation:** The retrieved text chunks are injected into a highly restrictive Gemini prompt (`temperature=0.1`) that forbids outside knowledge.
4. **Verification:** The generated response is embedded and mathematically compared to the retrieved chunks to calculate the hallucination probability.
