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
import Levenshtein as lev
llm = ChatOpenAI(model="gpt-3.5-turbo")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
import pandas as pd

# Define your desired data structure.
class GeocodingOutput(BaseModel):
    latitude: str = Field(description="latitude of the location")
    longitude: str = Field(description="longitude of the location")
    address: str = Field(description="address of the location")
    
parser = JsonOutputParser(pydantic_object=GeocodingOutput)

file_path = "./benchmark_dataset/splited/output_0.csv"

loader = CSVLoader(
    file_path=file_path,
)

import random

# Load the CSV file
df = pd.read_csv(file_path)

# Randomly sample 100 rows
sample_df = df.sample(n=100)

# Generate the questions
coordinate_questions = ["What is the coordinate of {}?".format(address) for address in sample_df['address']]
coordinate_answers = [{"latitude": lat, "longitude": long} for lat, long in zip(sample_df['latitude'], sample_df['longitude'])]

place_questions = ["Where is the place in {} {}?".format(lat, long) for lat, long in zip(sample_df['latitude'], sample_df['longitude'])]
place_answers = [address for address in sample_df['address']]

address_questions = ["Where is {}?".format(address) for address in sample_df['address']]
address_answers = [{"latitude": lat, "longitude": long, "address": address} for address, lat, long in zip(sample_df['address'], sample_df['latitude'], sample_df['longitude'])]

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

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

output_log = []


total_questions = len(place_questions)
total_score = 0
for place_question, place_answer in zip(place_questions, place_answers):
    output = rag_chain.invoke(place_question)
    lev_distance = lev.distance(output['address'], place_answer)
    similarity_score = 1 - lev_distance / max(len(output['address']), len(place_answer))
    total_score += similarity_score
    output_log.append(output)

accuracy_score = total_score / total_questions
print("Accuracy Score for Place Question", accuracy_score)


total_questions = len(coordinate_questions)
total_score = 0

for coordinate_question, coordinate_answer in zip(coordinate_questions, coordinate_answers):
    output = rag_chain.invoke(coordinate_question)
    
    output_coordinates = output['latitude'] + ', ' + output['longitude']
    answer_coordinates = str(coordinate_answer['latitude']) + ', ' + str(coordinate_answer['longitude'])
    
    lev_distance = lev.distance(output_coordinates, answer_coordinates)
    similarity_score = 1 - lev_distance / max(len(output_coordinates), len(answer_coordinates))
    
    total_score += similarity_score
    output_log.append(output)

accuracy_score = total_score / total_questions
print("Accuracy Score for Coordinate Question", accuracy_score)

with open('output_log.log', 'w') as f:
    for item in output_log:
        f.write("%s\n" % item)
