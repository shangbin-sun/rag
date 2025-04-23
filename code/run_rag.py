import os
import sys

import lazyllm
from lazyllm import (
    pipeline, 
    parallel, 
    bind, 
    OnlineEmbeddingModule, 
    Document, 
    Retriever, 
    Reranker
)

prompt = '你是一个AI助手，请根据用户的问题，从给定的上下文（context_str）中检索相关信息，并给出回答。'

documents = Document(
    dataset_path="/home/mnt/sunshangbin/work_dir/rag/code/"
)

documents.create_node_group(name="block", transform=(lambda d: d.split("。")))
with pipeline() as ppl:
    ppl.retriever = Retriever(
        documents, 
        group_name="CoarseChunk",  
        similarity="bm25_chinese", 
        topk=3,
        output_format='content',
        join=True   
    )
    
    ppl.llm = lazyllm.OnlineChatModule(
        stream=False
    ).prompt(lazyllm.ChatPrompter(prompt))

print(ppl("特朗普说了什么？"))