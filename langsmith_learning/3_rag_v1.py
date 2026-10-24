# pip install -U langchain langchain-openai langchain-community faiss-cpu pypdf python-dotenv

import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # expects OPENAI_API_KEY in .env

PDF_PATH = "islr.pdf"  # <-- change to your PDF filename

# 1) Load PDF
loader = PyMuPDFLoader(PDF_PATH)
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = OpenAIEmbeddings(
    model='local embeddinggemma-2 gguf',
    base_url='http://127.0.0.1:9932/v1',
    api_key='not needed'
)
for doc in splits:
    doc.page_content = f"title: none | text: {doc.page_content}"
    
vs = FAISS.from_documents(splits, emb)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatOpenAI(
    model='local qwen3.5',
    base_url='http://127.0.0.1:9931/v1',
    api_key='not needed'
)

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
os.environ['LANGCHAIN_PROJECT'] = 'RAG_demo'
config = {
    'tags': ['RAG_v1']
}

print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
question = input("\nQuestion: ")
ans_chunks = []

print('Answer: ')
for chunk in chain.stream(
    f"task: question answering | query: {question}"
):
    print(chunk, end='', flush=True)
    ans_chunks.append(chunk)

ans = ''.join(ans_chunks)
