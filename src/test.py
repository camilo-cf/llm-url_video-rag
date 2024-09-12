from fs.memoryfs import MemoryFS
from data.video_transcript import YTTranscript
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
# Define the web url
web_urls = "https://docs.smith.langchain.com/old/user_guide"
# Define the youtube video id
yt = ["jx7xuHlfsEQ"]
# Get the youtube transcript and join them
yt_transcripts = [YTTranscript(video).process() for video in yt]
yt_transcripts = "\n".join(yt_transcripts)

# Write the transcript in a temporary filsesystem
with mem_fs.open(video_transcript_filename, 'w') as f:
    f.write(yt_transcripts)

# Load all the RAGs
rag_dict = {
    "SimpleRAGVectorSearch": SimpleRAGVectorSearch(web_urls, video_transcript_filename, mem_fs),
    "RerankRAGVectorSearch": RerankRAGVectorSearch(web_urls, video_transcript_filename, mem_fs),
    "SimpleRAGKeywordSearch": SimpleRAGKeywordSearch(web_urls, video_transcript_filename, mem_fs),
    "RerankRAGKeywordSearch": RerankRAGKeywordSearch(web_urls, video_transcript_filename, mem_fs),
    "SimpleRAGEnsembleVectorKeywordSearch": SimpleRAGEnsembleVectorKeywordSearch(web_urls, video_transcript_filename, mem_fs),
    "RerankRAGEnsembleVectorKeywordSearch": RerankRAGEnsembleVectorKeywordSearch(web_urls, video_transcript_filename, mem_fs),
}

# Rewrite query/ question
query = "How can langsmith help me with testing?"
query = query_rewrite(query)
print("Rewritten Query:", query)

# Test all the RAGs
for rag_name, rag in rag_dict.items():
    print("################################################")
    print("Model: ", rag_name)
    response = rag.augment_generate(query)
    print("Response:\n", response)
    print("################################################")
