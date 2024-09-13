from fs.memoryfs import MemoryFS
from data.data_interface import URLHandler
from data.video_transcript import YTTranscript
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from io import StringIO

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

print("YT:", yt)
print("WEB", web_urls)

# Get the youtube transcript and join them
yt_transcripts = [YTTranscript(video).process() for video in yt]
yt_transcripts = "\n".join(yt_transcripts)

# Write the transcript in a temporary filsesystem
with mem_fs.open(video_transcript_filename, 'w') as f:
    f.write(yt_transcripts)

rag = SimpleRAGVectorSearch(web_urls, video_transcript_filename, mem_fs)
docs = rag.docs
split_documents = rag.split_documents


import os
os.environ["OPENAI_API_KEY"] = "API-KEY"
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

GENERATE_QUESTIONS = """
You emulate an user who is using a QA for some given CONTEXT documents and video trnascriptions.
Please formulate 5 questions and answers an user might ask and should get as answer\
based on the CONTEXT, the questions should be complete and not too short.
If possible, use as fewer words as possible from the CONTEXT.
Answer the questions as well based on the CONTEXT.

CONTEXT: {context}
QUESTION: <question>
ANSWER: <answer>

Provide the output in a parsable TSV without using any code blocks:

question_title\t answer_title\n
question1\t answer1\n
question2\t answer2\n
...
question5\t answer5\n
"""

def generate_questions(context):
    prompt = ChatPromptTemplate.from_template(GENERATE_QUESTIONS)
    chain = prompt | llm
    return chain.invoke(context)
    
df_list =[]
for each in split_documents:
    questions = generate_questions(each.page_content)
    content = questions.content

    # Use StringIO to simulate reading from a file
    data_io = StringIO(content)

    df = pd.read_csv(data_io, sep='\t')
    df["source"] = each.json()
    df_list.append(df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(df_list, ignore_index=True)

# Display the final DataFrame
print(final_df)

final_df.to_pickle('goundtruth_dataset.pkl')