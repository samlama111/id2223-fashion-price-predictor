import os
import google.generativeai as genai

# This and the embed_text function are taken from https://ai.google.dev/gemini-api/docs/embeddings#generate-embeddings
genai.configure(api_key=os.environ["GOOGLE_AI_STUDIO_API_KEY"])

def embed_text(text):
    # Embeddings are free, as defined at https://ai.google.dev/pricing#text-embedding004
    result = genai.embed_content(
            model="models/text-embedding-004",
            content=text)

    return result['embedding']