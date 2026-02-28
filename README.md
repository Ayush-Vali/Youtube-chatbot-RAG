# Youtube 

For this project, the transcript can be loaded using 
1) YTLoader 
2) Youtube API\
Since YTLoader can be buggy, We'll choose 2nd option



### Retriever
- it acts as Runnable

from langchain_community.vectorstores import FAISS\
vector_store = FAISS.from_documents(chunks, embeddings)\
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})\
retriever.invoke('What is deepmind')
