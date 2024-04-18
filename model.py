from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

VECTOR_STORE_PATH = 'vector_store/embedding_store'

custom_prompt = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def initialize_custom_prompt():
    """
    Define the prompt template for QA retrieval from the vector store
    """
    prompt_template = PromptTemplate(template=custom_prompt,
                                     input_variables=['context', 'question'])
    return prompt_template

# Create Retrieval QA Chain
def build_retrieval_qa_chain(llm, prompt_template, vector_store):
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm,
                                                  chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                                  return_source_documents=True,
                                                  chain_type_kwargs={'prompt': prompt_template}
                                                  )
    return retrieval_chain

# Load the model
def load_language_model():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=256,
        temperature=0.5
    )
    return llm

# QA Bot Function
def question_answer_bot():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                            model_kwargs={'device': 'cpu'})
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    llm = load_language_model()
    qa_prompt_template = initialize_custom_prompt()
    qa_bot = build_retrieval_qa_chain(llm, qa_prompt_template, vector_store)

    return qa_bot

# Output function
def generate_response(query):
    qa_chain = question_answer_bot()
    response = qa_chain({'query': query})
    return response

# Chainlit code
@cl.on_chat_start
async def initiate_chat():
    qa_chain = question_answer_bot()
    initial_message = cl.Message(content="Initializing the bot...")
    await initial_message.send()
    initial_message.content = "Hi, Welcome to VIT Bot. What is your query?"
    await initial_message.update()

    cl.user_session.set("qa_chain", qa_chain)

@cl.on_message
async def handle_message(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    callback_handler = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True
    response = await qa_chain.acall(message.content, callbacks=[callback_handler])
    final_answer = response["result"]

    await cl.Message(content=final_answer).send()
