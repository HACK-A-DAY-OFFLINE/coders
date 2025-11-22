from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load text from a TXT file
def load_text_from_txt(txt_file):
    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# 2. Split content into sentences / bullet points
def split_into_sentences(text):
    import re
    # supports bullet points and numbers
    lines = re.split(r'\n|•|- |\*|\d+\)', text)
    sentences = [l.strip() for l in lines if l.strip()]
    return sentences

# 3. Build knowledge base
def build_knowledge_base(txt_file):
    text = load_text_from_txt(txt_file)
    sentences = split_into_sentences(text)
    vectorizer = TfidfVectorizer().fit(sentences)
    sentence_vectors = vectorizer.transform(sentences)
    return sentences, vectorizer, sentence_vectors

# 4. Answer question
def answer_question(question, sentences, vectorizer, sentence_vectors, top_k=1):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, sentence_vectors).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    best_sentence = sentences[top_indices[0]]
    return f"\n📌 Most Relevant Answer:\n{best_sentence}"

# ---------- DEMO ----------
if __name__ == "__main__":
    # change filename here
    sentences, vectorizer, sentence_vectors = build_knowledge_base("notes.txt")

    print("\n🧠 Offline TXT Chatbot Ready! (type 'exit' to quit)")
    while True:
        query = input("\n❓ You: ")
        if query.lower() == "exit":
            break
        answer = answer_question(query, sentences, vectorizer, sentence_vectors, top_k=1)
        print("🤖 Bot:", answer)