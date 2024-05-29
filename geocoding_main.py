import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add LANGCHAIN_API_KEY to environment variables
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT']="llm-geocoding-v1"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define your desired data structure.
class GeocodingOutput(BaseModel):
    latitude: str = Field(description="latitude of the location")
    longitude: str = Field(description="longitude of the location")
    address: str = Field(description="address of the location")
    
parser = JsonOutputParser(pydantic_object=GeocodingOutput)

loader = CSVLoader(
    file_path="./benchmark_dataset/splited/sample.csv",
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
print("Vectorstore created")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

ChatPromptTemplate = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
{format_instructions}
Question: {question} 
Context: {context} 
Answer:
"""

prompt = PromptTemplate(
    template=ChatPromptTemplate,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

output = rag_chain.invoke("Where is Mandan Police Department 49900?")

print(output)