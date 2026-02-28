from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = Flask(__name__)

# ====================== GLOBAL (simple for Render free tier) ======================
vector_store = None
qa_chain = None

# LLM & Embeddings (exactly as in your notebook)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.7,
    max_new_tokens=512,
)
llm = ChatHuggingFace(llm=llm)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

prompt = PromptTemplate(
    template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ====================== HELPER FUNCTIONS ======================
def build_rag(video_id: str):
    global vector_store, qa_chain

    # 1. Fetch transcript
    transcript_list = YouTubeTranscriptApi().fetch(video_id)
    full_transcript = " ".join([item.text for item in transcript_list])

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([full_transcript])

    # 3. Embed + FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 4. Chain (exactly your final chain)
    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    qa_chain = parallel | prompt | llm | StrOutputParser()

    # Simple summary (you can improve this later)
    summary = full_transcript[:800] + "..." if len(full_transcript) > 800 else full_transcript
    return full_transcript, summary

# ====================== ROUTES ======================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global qa_chain
    url_or_id = request.form.get('video_id').strip()

    # Extract video ID if full URL
    if 'youtu' in url_or_id:
        video_id = url_or_id.split('v=')[1]
    else:
        video_id = url_or_id

    try:
        transcript, summary = build_rag(video_id)
        return jsonify({
            "status": "success",
            "video_id": video_id,
            "transcript_preview": transcript[:2000] + "..." if len(transcript) > 2000 else transcript,
            "summary": summary[:600] + "..." if len(summary) > 600 else summary
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/ask', methods=['POST'])
def ask():
    if qa_chain is None:
        return jsonify({"status": "error", "message": "Process a video first!"})
    
    question = request.form.get('question').strip()
    try:
        answer = qa_chain.invoke(question)
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)