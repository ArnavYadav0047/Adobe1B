import os
import json
import time
import fitz  
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("model/all-MiniLM-L6-v2")



def extract_text_chunks(pdf_path):
    """Extract paragraphs from each page of the PDF."""
    doc = fitz.open(pdf_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue

        
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 30]

        for para in paragraphs:
            chunks.append({
                "document": os.path.basename(pdf_path),
                "page": page_num,
                "text": para
            })

    return chunks


def rank_chunks(chunks, query, top_k=10):
    """Embed and rank text chunks based on similarity to the query."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = scores.topk(k=top_k)

    ranked = []
    for score, idx in zip(top_results[0], top_results[1]):
        chunk = chunks[int(idx)]
        chunk["score"] = float(score)
        ranked.append(chunk)

    return ranked


def process(input_dir, persona, job, output_path):
    start_time = time.time()
    all_chunks = []

    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            all_chunks.extend(extract_text_chunks(pdf_path))

    query = f"{persona}. {job}"
    top_chunks = rank_chunks(all_chunks, query, top_k=10)

    
    result = {
        "metadata": {
            "input_documents": list(set([c["document"] for c in all_chunks])),
            "persona": persona,
            "job_to_be_done": job,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        },
        "extracted_sections": [
            {
                "document": c["document"],
                "page_number": c["page"],
                "section_title": c["text"][:60] + "...",
                "importance_rank": i + 1
            }
            for i, c in enumerate(top_chunks)
        ],
        "sub_section_analysis": [
            {
                "document": c["document"],
                "page_number": c["page"],
                "refined_text": c["text"]
            }
            for c in top_chunks
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Processed {len(all_chunks)} chunks in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    persona = open("/app/data/persona.txt", encoding="utf-8").read().strip()
    job = open("/app/data/job_description.txt", encoding="utf-8").read().strip()
    process("/app/input", persona, job, "/app/output/result.json")
