from fs.memoryfs import MemoryFS
from data.data_interface import URLHandler
from data.video_transcript import YTTranscript
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

from rag.query_rewrite import query_rewrite
from rag.rag import (
    SimpleRAGVectorSearch,
    RerankRAGVectorSearch,
    SimpleRAGKeywordSearch,
    RerankRAGKeywordSearch,
    SimpleRAGEnsembleVectorKeywordSearch,
    RerankRAGEnsembleVectorKeywordSearch
    )

# Start a temporary filesystem
mem_fs = MemoryFS()
# Define the video transcript filename
video_transcript_filename = "video_transcripts.txt"

# Input URLs to measure on
urls = """
https://en.wikipedia.org/wiki/LangChain\n
https://en.wikipedia.org/wiki/Generative_pre-trained_transformer\n
https://en.wikipedia.org/wiki/OpenAI\n
https://en.wikipedia.org/wiki/Auto-GPT\n
https://www.youtube.com/watch?v=mTnQZn3ZVyI\n
https://www.youtube.com/watch?v=jpIskGUQPyI&pp=ygULb3BlbmFpIG5ld3M%3D\n
"""

handled_urls = URLHandler(urls=urls)
yt = handled_urls.youtube
web_urls = handled_urls.website

# Get the youtube transcript and join them
yt_transcripts = [YTTranscript(video).process() for video in yt]
yt_transcripts = "\n".join(yt_transcripts)

# Write the transcript in a temporary filsesystem
with mem_fs.open(video_transcript_filename, 'w') as f:
    f.write(yt_transcripts)


df = pd.read_pickle('src/groundtruth_dataset.pkl')

# Load all the RAGs
rag_dict = {
    "SimpleRAGVectorSearch": SimpleRAGVectorSearch(web_urls, video_transcript_filename, mem_fs),
    "RerankRAGVectorSearch": RerankRAGVectorSearch(web_urls, video_transcript_filename, mem_fs),
    "SimpleRAGKeywordSearch": SimpleRAGKeywordSearch(web_urls, video_transcript_filename, mem_fs),
    "RerankRAGKeywordSearch": RerankRAGKeywordSearch(web_urls, video_transcript_filename, mem_fs),
    "SimpleRAGEnsembleVectorKeywordSearch": SimpleRAGEnsembleVectorKeywordSearch(web_urls, video_transcript_filename, mem_fs),
    "RerankRAGEnsembleVectorKeywordSearch": RerankRAGEnsembleVectorKeywordSearch(web_urls, video_transcript_filename, mem_fs),
}

embeddings = OllamaEmbeddings(
            # model="nomic-embed-text"
            model="mxbai-embed-large",
            )

def text_embedding(text):
    return np.asarray(embeddings.embed_query(text))

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

text_embedding("hola")

def similarity_calculation(df, rag):
    total_similarity = []
    for index, row in df.iterrows():
        retrieve_response = rag.augment_generate(row["question_title"])
        expected_answer = df["answer_title"][index]

        response_embedding = text_embedding(retrieve_response)
        expected_embedding = text_embedding(expected_answer)

        similarity = cosine_similarity(response_embedding, expected_embedding)
        total_similarity.append(similarity)

    return total_similarity


rag_dict_similarity = {}
# Test all the RAGs
print("################################################")
for rag_name, rag in rag_dict.items():
    print("Model: ", rag_name)
    similarity = similarity_calculation(df, rag)
    avg_similarity = np.asarray(similarity).mean()
    print("Average Expected Answer vs RAG value:\n", avg_similarity)
    print("################################################")
    rag_dict_similarity[rag_name] = avg_similarity

# Convert the similarity to a pandas DataFrame
avg_similarity_df = pd.DataFrame(list(rag_dict_similarity.items()), columns=['Model', 'Similarity'])

# Plot the similarity
plt.figure(figsize=(10, 6))
plt.bar(avg_similarity_df['Model'], avg_similarity_df['Similarity'], color='skyblue')

# Add labels and title
plt.xlabel('RAG Model')
plt.ylabel('Similarity')
plt.title('Similarity for Each RAG Model Retrieval Algorithm')
plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()

# Display the plot
plt.show()
