from sentence_transformers import SentenceTransformer

# Load from HuggingFace (only needs to be done once online)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Save the full model to your folder
model.save('model/all-MiniLM-L6-v2')
