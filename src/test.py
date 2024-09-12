from fs.memoryfs import MemoryFS
from data.video_transcript import YTTranscript
from rag.rag import RAG

mem_fs = MemoryFS()
video_transcript_filename = "video_transcripts.txt"

web_urls = "https://docs.smith.langchain.com/old/user_guide"
yt = ["jx7xuHlfsEQ"]

yt_transcripts = [YTTranscript(video).process() for video in yt]
yt_transcripts = "\n".join(yt_transcripts)

# Write a file   
with mem_fs.open(video_transcript_filename, 'w') as f:
    f.write(yt_transcripts)

rag = RAG(web_urls, video_transcript_filename, mem_fs)
query = "How can langsmith help me with testing?"

response = rag.augment_generate(query)
print(response)