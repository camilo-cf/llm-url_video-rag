from fs.memoryfs import MemoryFS
from data.data_interface import URLHandler
from data.video_transcript import YTTranscript
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from io import StringIO
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

def hit_rate_calculation(df, rag):
    hit_rate_counter = 0
    for index, row in df.iterrows():
        retrieve_response = rag.simple_retriever.invoke(row["question_title"])
        for each in retrieve_response:
            if df["source"][index] == each.json():            
                hit_rate_counter += 1

    hit_rate = hit_rate_counter/ len(df)
    return hit_rate


rag_dict_hit_rate = {}
# Test all the RAGs
print("################################################")
for rag_name, rag in rag_dict.items():
    print("Model: ", rag_name)
    hit_rate = hit_rate_calculation(df, rag)
    print("Hit Rate:\n", hit_rate)
    print("################################################")
    rag_dict_hit_rate[rag_name] = hit_rate

# Convert the hit rates to a pandas DataFrame
hit_rate_df = pd.DataFrame(list(rag_dict_hit_rate.items()), columns=['Model', 'Hit Rate'])

# Plot the hit rates
plt.figure(figsize=(10, 6))
plt.bar(hit_rate_df['Model'], hit_rate_df['Hit Rate'], color='skyblue')

# Add labels and title
plt.xlabel('RAG Model')
plt.ylabel('Hit Rate')
plt.title('Hit Rate for Each RAG Model Retrieval Algorithm')
plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()

# Display the plot
plt.show()
