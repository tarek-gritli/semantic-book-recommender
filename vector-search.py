from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
books = pd.read_csv('books_cleaned.csv')
books["tagged_description"].to_csv("tagged_descriptions.txt", index=False, sep="\n", header=False)

raw_documents = TextLoader("tagged_descriptions.txt").load()

with open("tagged_descriptions.txt", "r", encoding="utf-8") as f:
    documents = [Document(page_content=line.strip()) for line in f if line.strip()]

print(len(documents))

db_books = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

def retrieve_semantic_recommendations(query, k=10):
    results = db_books.similarity_search(query, k=k)
    
    books_list = []
    
    for i in range (len(results)):
        books_list += [int(results[i].page_content.strip('"').split()[0])]
        
    return books[books["isbn13"].isin(books_list)]


print(retrieve_semantic_recommendations("A book to teach children about nature"))
