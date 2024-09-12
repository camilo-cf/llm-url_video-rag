from langchain_community.document_loaders import MergedDataLoader
from langchain.schema import Document
from fs.memoryfs import MemoryFS
from langchain.document_loaders.base import BaseLoader

# Custom InMemoryLoader class
class InMemoryLoader(BaseLoader):
    def __init__(self, mem_fs, transcript_filename):
        self.mem_fs = mem_fs
        self.transcript_filename = transcript_filename

    def load(self):
        # Open the in-memory file for reading
        with self.mem_fs.open(self.transcript_filename, 'r') as file:
            content = file.read()

        # Return a LangChain Document with the content
        return [Document(page_content=content)]

if __name__ == "__main__":
    # Example MemoryFS setup
    mem_fs = MemoryFS()
    video_transcript_filename = "video_transcripts.txt"

    # Write sample content to the in-memory file
    with mem_fs.open(video_transcript_filename, 'w') as f:
        f.write("This is a test transcript for a YouTube video.")

    # Create the InMemoryLoader instance
    in_memory_loader = InMemoryLoader(mem_fs, video_transcript_filename)
    documents = in_memory_loader.load()

    print(documents)

    # Create another loader (for merging, you can add more loaders as needed)
    # For demonstration purposes, we'll reuse the same InMemoryLoader instance
    other_loader = InMemoryLoader(mem_fs, video_transcript_filename)

    # Use MergedDataLoader to merge content from multiple loaders
    merged_loader = MergedDataLoader([in_memory_loader, other_loader])

    # Load the merged content
    documents = merged_loader.load()

    # Output the merged documents
    print(documents)