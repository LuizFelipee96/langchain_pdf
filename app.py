from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Faça alguma pergunta ao artigo")
    st.header("Pergunte para o PDF")
    
    #Carregar Arquivo
    pdf = st.file_uploader("Carregue seu artigo em PDF", type="pdf")
    
    #Extrair o Texto
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

    #Divisor de texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        st.write(chunks)
        
    #Criar embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
    #Caixa de Arquivo para introduzir a pergunta do usuario
        userquestion = st.text_input("Faça alguma pergunta em questão do seu interesse: ")
        if userquestion:
            docs = knowledge_base.similarity_search(userquestion)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=userquestion)
                print(cb)
            st.write(response)
            
        
if __name__ == '__main__':
    main()