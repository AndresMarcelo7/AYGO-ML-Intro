# AYGO-ML-Intro


## Workshop

Using Lang Chain and Pinecone develop the following challenges:

1. From Python write a program to send prompts to Chatgpt and retrieve responses.[1_basic_chat.py](1_basic_chat.py)
2. Write a simple RAG using an in-memory vector database.[2_simple_rag.py](2_simple_rag.py)
3. Write a RAG using Pinecone.[3_pinecone_rag.py](3_pinecone_rag.py)

## How to run?
Specify the required env variables followed by ```python [SCRIPT]```

Example:

```shell
PINECONE_API_KEY=[YOURPINECONEAPIKEY] OPENAI_API_KEY=[YOUROPENAIAPIKEY] python3 3_pinecone_rag.py
```

## Explanation 

### 1. From Python write a program to send prompts to Chatgpt and retrieve responses.

In this code we are using langChani Library to interact with OpenAI

First we define a specific Prompt that the LLM will use and then, we use Langchain's LLM to make calls to OpenAI Service using the API KEY.
```python
llm = OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

response = llm_chain.run(question)
```


### 2.Write a simple RAG using an in-memory vector database.

In this exercise we load a file from the web [file]("https://lilianweng.github.io/posts/2023-06-23-agent/") using ```WebBaseLoader``` and then 
we Split the loaded documents into chunks using ```RecursiveCharacterTextSplitter```.

```python
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

```

The next step is to Vectorize the document chunks using ```Chroma``` and ```OpenAI``` embeddings, and creating a retriever for further use,
this step is very important because it has the purpose of transforming textual data (document) into numerical vectors that capture semantic information about the content.
```python
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

then we retrieve a prompt from Langchain's hub and initialize a specific ChatOpenAI model.
```python
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

In the next step we define a chain that specifies the vector, model, prompt and output that the we will use to ask our question, we invoke it and we have our response. 
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is Task Decomposition?")
```

### 3.Write a RAG using Pinecone

In this exersice we save the vectorized text in Pinecone and consume it every question.

We have to functions ```loadText()``` and ```search()```.

In ```loadText()``` we load an [exampleText.txt](exampleText.txt) split it into chunks and load it.

then, we use Pinecone to check if theres any index stored (Any vectorized text), if not, we create one
```python
    import pinecone
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
    )
    index_name = "langchain-demo"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
```

In ```search()``` we initialize Pinecone and then we use the index name to query directly with PineCone
```python

    index_name = "langchain-demo"
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    query = "give me the references of the paper"
    docs = docsearch.similarity_search(query)

    print(docs[0].page_content)
```

Then we print the result.

## Conclusion

This workshop was very usefull because I learned some utilities to interact with OpenAI models using python, also
I learned to use a workflow for loading text documents, processing them using OpenAI embeddings, and utilizing local storage and Pinecone for vector storage and similarity search. 



