from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chains import RetrievalQA
from redundant_filter_retriever import RedundantFilterRetriever
import langchain

langchain.debug = True

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "EleutherAI/gpt-neo-1.3B" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1000,
    max_new_tokens=200 )
chat = HuggingFacePipeline(pipeline=pipe)

# Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("How to make a call?")
print(result)
