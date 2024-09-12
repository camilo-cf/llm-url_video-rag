from fs.memoryfs import MemoryFS
import gradio as gr
import time

from data.data_interface import URLHandler
from data.video_transcript import YTTranscript
from rag.query_rewrite import query_rewrite
from rag.rag import RerankRAGEnsembleVectorKeywordSearch

mem_fs = MemoryFS()
video_transcript_filename = "video_transcripts.txt"

def load_urls(urls):
    handled_urls = URLHandler(urls=urls)
    yt = handled_urls.youtube
    web = handled_urls.website

    global yt_transcripts
    yt_transcripts = [YTTranscript(video).process() for video in yt]
    yt_transcripts = "\n".join(yt_transcripts)

    # Write a file   
    with mem_fs.open(video_transcript_filename, 'w') as f:
        f.write(yt_transcripts)

    global rag
    rag = RerankRAGEnsembleVectorKeywordSearch(web, video_transcript_filename, mem_fs, "llama3.1")

    return "Processing complete! ✅"


def rag_fn(message, history):
    query = query_rewrite(message)
    print(query)
    response = rag.augment_generate(query)
    return response


with gr.Blocks() as app:

    gr.Markdown(
    """
    # ✨ Chat with Your YouTube Video Transcripts & Favorite URLs! 🎥💻

    ## Interact 🤝, Explore 🌍, and Learn 📚 from the content you love ❤️

    *🌍 URLs and Interactions should be **only** English 🤝*

    *🎥 YouTube videos can be used if the **English transcription is available**🎥*
    """)

    urls = gr.Textbox("https://docs.smith.langchain.com/old/user_guide \nhttps://www.youtube.com/watch?v=jx7xuHlfsEQ",
                      label="Enter the YouTube video URLs or website URLs separated by new lines")
    
    wait_message = gr.Markdown("⌛ Please wait, processing...", visible=False)
    
    button = gr.Button("Load URLs")
    
    result = gr.Textbox(label="Result", visible=False)

    with gr.Column(visible=False) as chat_column:
        chat_interface = gr.ChatInterface(fn=rag_fn,
                                          examples=["How can langsmith help me with testing?", "What is langsmith?"],
                                          title="🎥💻 URL Chat Bot 🌍✨")

    def handle_click(urls):
        # Hide the URL textbox, show the wait message, and make the chat interface visible
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

    button.click(fn=handle_click, inputs=[urls], outputs=[urls, wait_message, result, chat_column])
    button.click(fn=load_urls, inputs=[urls], outputs=[result])

    gr.Markdown(
    """
    *Ollama Powered RAG*

    By: [camilo-cf](https://github.com/camilo-cf/) with ⚡🚀❤️
    """)

app.launch(show_api=False)
