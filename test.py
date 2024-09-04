from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
from dotenv import dotenv_values
config = dotenv_values(".env")
import torch

def models(device):
    
    dataFamily = f"./Data/"
    vectordb = Chroma(persist_directory=dataFamily, embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()

    # Init Model
    model_path="openthaigpt/openthaigpt-1.0.0-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    model.to(device)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=520, temperature=0.1)
    llm = HuggingFacePipeline(pipeline=pipe)

    system_template = """
    system: คุณคือผู้ช่วยที่ตอบคำถามภายในสถาบันเทคโนโลยีจิตรลดาของเรา คุณจะตอบคถามที่มีประโยชน์ต่อนักศึกษามากที่สุด และ เมื่อสิ้นสุดประโยคคุณต้องลงท้ายด้วย ``ค่ะ``\n
    """

    Template = """
    กรุณาตอบตาม Context นี้เพื่อสร้างคำตอบที่มีประโยชน์สูงสุดให้แก่ นักศึกษาของเรา: {context}

    Question: {question}
    """

    template = system_template + Template

    prompt = ChatPromptTemplate.from_template(template)

    setup_and_retrieval = RunnableParallel(
        {"context": retriever,
        "question": RunnablePassthrough()}
        )
    output_parser = StrOutputParser()
    planer_chain = setup_and_retrieval | prompt | llm | output_parser
    return planer_chain



if __name__ == "__main__":
    # Ensure CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # res = model(device)

    # model_path="openthaigpt/openthaigpt-1.0.0-7b-chat"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    # model.to(device)

    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=520, temperature=0)
    # llm = HuggingFacePipeline(pipeline=pipe)

    # Prompt
    res =  models(device)
    prompt = str(input("กรุณาถามคำถาม: "))

    # while True:
    #     prompt = str(input("\nกรุณากรอกคำถาม: "))
    # llama_prompt = f"<s>[INST] <<SYS>>\nYou are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด<</SYS>>\n\n{prompt} [/INST]\n"
    for chunks in res.stream(prompt):
        print(chunks, end="", flush=True)





