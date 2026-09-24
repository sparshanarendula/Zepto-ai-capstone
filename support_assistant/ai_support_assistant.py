import os
from typing import List
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

##############################
# 1. PROJECT SETTINGS
##############################

#folder containing the 8 Zepto policy documents

POLICY_FOLDER = "zepto_policies"

#folder where ChromaDB will store the vector database

CHROMA_FOLDER = "zepto_chroma_db"

#name of the embedding model

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#name of the free local LLM

LLM_MODEL = "llama3.2"

##############################
# 2. CHECK POLICY DOCUMENTS
##############################

required_documents = [
    "delivery.txt",
    "returns_refunds.txt",
    "membership.txt",
    "order_tracking.txt",
    "cancellation.txt",
    "damaged_missing_items.txt",
    "gift_cards.txt",
    "support_hours.txt"
]
print("\n======================================")
print("CHECKING ZEPTO POLICY DOCUMENTS")
print("======================================")

if not os.path.exists(POLICY_FOLDER):
    os.makedirs(POLICY_FOLDER)
    print("\nPolicy folder created:")
    print(POLICY_FOLDER)
    print("\nPlease add the 8 Zepto policy documents into this folder.")
    print("Then run the program again.")

    exit()

missing_documents = []

for document_name in required_documents:
    document_path = os.path.join(
        POLICY_FOLDER,
        document_name
    )
    if not os.path.exists(document_path):
        missing_documents.append(document_name)
    if len(missing_documents) > 0:
        print("\nMissing policy documents:")
    for document_name in missing_documents:
        print("-", document_name)
    print("\nPlease add all 8 policy documents and run again.")

    exit()

    print("\nAll 8 policy documents found.")

##############################
# 3. LOAD POLICY DOCUMENTS
##############################

documents = []


print("\n======================================")
print("LOADING POLICY DOCUMENTS")
print("======================================")

for document_name in required_documents:
    document_path = os.path.join(
        POLICY_FOLDER,
        document_name
    )
    with open(
        document_path,
        "r",
        encoding="utf-8"
    ) as file:
        document_text = file.read()

documents.append(
        Document(
            page_content=document_text,
            metadata={
                "source": document_name
            }
        )
    )

print("\nTotal policy documents loaded:")
print(len(documents))

#install the required libraries in VS Code terminal

#pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama chromadb sentence-transformers pydantic


#import the required libraries

import os

from typing import List

from pydantic import BaseModel, Field

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama

from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter


##############################
# 1. PROJECT SETTINGS
##############################

#folder containing the 8 Zepto policy documents

POLICY_FOLDER = "zepto_policies"

#folder where ChromaDB will store the vector database

CHROMA_FOLDER = "zepto_chroma_db"

#name of the embedding model

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#name of the free local LLM

LLM_MODEL = "llama3.2"


##############################
# 2. CHECK POLICY DOCUMENTS
##############################

required_documents = [

    "delivery.txt",

    "returns_refunds.txt",

    "membership.txt",

    "order_tracking.txt",

    "cancellation.txt",

    "damaged_missing_items.txt",

    "gift_cards.txt",

    "support_hours.txt"

]


print("\n======================================")

print("CHECKING ZEPTO POLICY DOCUMENTS")

print("======================================")


if not os.path.exists(POLICY_FOLDER):

    os.makedirs(POLICY_FOLDER)

    print("\nPolicy folder created:")

    print(POLICY_FOLDER)

    print("\nPlease add the 8 Zepto policy documents into this folder.")

    print("Then run the program again.")

    exit()


missing_documents = []


for document_name in required_documents:

    document_path = os.path.join(

        POLICY_FOLDER,

        document_name

    )

    if not os.path.exists(document_path):

        missing_documents.append(document_name)


if len(missing_documents) > 0:

    print("\nMissing policy documents:")

    for document_name in missing_documents:

        print("-", document_name)

    print("\nPlease add all 8 policy documents and run again.")

    exit()


print("\nAll 8 policy documents found.")


##############################
# 3. LOAD POLICY DOCUMENTS
##############################

documents = []


print("\n======================================")

print("LOADING POLICY DOCUMENTS")

print("======================================")


for document_name in required_documents:

    document_path = os.path.join(

        POLICY_FOLDER,

        document_name

    )

    with open(

        document_path,

        "r",

        encoding="utf-8"

    ) as file:

        document_text = file.read()


    documents.append(

        Document(

            page_content=document_text,

            metadata={

                "source": document_name

            }

        )

    )


print("\nTotal policy documents loaded:")

print(len(documents))


##############################
# 4. SPLIT DOCUMENTS INTO CHUNKS
##############################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("\n======================================")
print("DOCUMENT CHUNKING")
print("======================================")
print("\nTotal chunks created:")
print(len(chunks))

##############################
# 5. CREATE HUGGING FACE EMBEDDINGS
##############################

print("\n======================================")
print("CREATING EMBEDDINGS")
print("======================================")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("\nHugging Face embedding model loaded successfully.")

##############################
# 6. CREATE CHROMADB VECTOR DATABASE
##############################

print("\n======================================")
print("CREATING CHROMADB VECTOR DATABASE")
print("======================================")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_FOLDER,
    collection_name="zepto_policies"
)

print("\nChromaDB vector database created successfully.")

##############################
# 7. CREATE RETRIEVER
##############################

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)

##############################
# 8. CREATE FREE LOCAL LLM
##############################

print("\n======================================")
print("LOADING LOCAL LLM")
print("======================================")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

print("\nLocal LLM loaded:")
print(LLM_MODEL)

##############################
# 9. STRUCTURED RESPONSE FORMAT
##############################

class ChatbotResponse(BaseModel):
    answer: str = Field(
        description="Clear answer to the customer's question based only on the Zepto policy documents."
    )
    policy_area: str = Field(
        description="Policy area related to the question."
    )
    source_documents: List[str] = Field(
        description="Names of the policy documents used to answer the question."
    )

    confidence: str = Field(
        description="High, Medium, or Low based on how clearly the policy documents answer the question."
    )

structured_llm = llm.with_structured_output(
    ChatbotResponse
)

##############################
# 10. CREATE CHATBOT FUNCTION
##############################

def zepto_chatbot(user_question):
    #retrieve relevant policy documents
    retrieved_documents = retriever.invoke(
        user_question
    )

    #checking whether documents were retrieved
    if not retrieved_documents:
        return ChatbotResponse(
            answer="I could not find relevant information in the available Zepto policy documents.",
            policy_area="Unknown",
            source_documents=[],
            confidence="Low"
        )

    #combining retrieved document content
    context = "\n\n".join(
        [
            f"Source: {document.metadata.get('source', 'Unknown')}\n"
            f"Content:\n{document.page_content}"
            for document in retrieved_documents
        ]
    )

    #getting unique source document names
    source_documents = list(
        dict.fromkeys(
            [
                document.metadata.get(
                    "source",
                    "Unknown"
                )
                for document in retrieved_documents
            ]
        )
    )

    #creating prompt for the LLM
    prompt = f"""
You are a Zepto customer support chatbot.
Answer the customer's question using ONLY the policy information provided below.
Do not invent information.
Do not use general knowledge if the answer is not present in the policy documents.
If the policy documents do not contain enough information, clearly say that the information is not available in the provided policies.
Give a concise and customer-friendly answer.

Customer Question:
{user_question}

Retrieved Zepto Policy Information:
{context}

Important instructions:
1. Answer only from the retrieved policy information.
2. Identify the relevant policy area.
3. Mention the policy documents used.
4. Set confidence to High when the answer is clearly supported.
5. Set confidence to Medium when the answer is partially supported.
6. Set confidence to Low when the policy information is insufficient.
"""

    #generate structured response
    response = structured_llm.invoke(
        prompt
    )

    #make sure retrieved source documents are included
    response.source_documents = source_documents

    return response

##############################
# 11. TEST QUESTIONS
##############################

print("\n======================================")
print("TESTING ZEPTO CHATBOT")
print("======================================")

test_questions = [
    "What should I do if my order is delayed?",
    "Can I cancel my order?",
    "What should I do if an item is missing?",
    "How can I track my order?",
    "What are the support hours?"
]

for question in test_questions:
    print("\n--------------------------------------")
    print("Customer Question:")
    print(question)

    response = zepto_chatbot(
        question
    )

    print("\nChatbot Answer:")
    print(response.answer)
    print("\nPolicy Area:")
    print(response.policy_area)
    print("\nSource Documents:")
    print(response.source_documents)
    print("\nConfidence:")
    print(response.confidence)

##############################
# 12. INTERACTIVE CHATBOT
##############################

print("\n======================================")
print("ZEPTO AI CUSTOMER SUPPORT CHATBOT")
print("======================================")
print("\nType 'exit' to stop the chatbot.")

while True:
    user_question = input(
        "\nCustomer: "
    )

    if user_question.lower().strip() == "exit":
        print("\nThank you for using Zepto Customer Support Chatbot.")
        break

    if user_question.strip() == "":
        print("\nPlease enter a question.")
        continue

    response = zepto_chatbot(
        user_question
    )

    print("\nZepto Chatbot:")
    print(response.answer)
    print("\nPolicy Area:")
    print(response.policy_area)
    print("\nSources:")

    for source in response.source_documents:
        print("-", source)

    print("\nConfidence:")
    print(response.confidence)

##############################
# 13. PROJECT COMPLETION
##############################

print("\n======================================")
print("PROJECT COMPLETED")
print("======================================")
print("8 Zepto policy documents used.")
print("RAG implemented using ChromaDB.")
print("Embeddings generated using Hugging Face Sentence Transformers.")
print("Local open-source LLM used:", LLM_MODEL)
print("LangChain used for orchestration.")
print("Structured output implemented using Pydantic.")
print("Interactive customer support chatbot created.")