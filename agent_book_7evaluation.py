from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai
from datetime import datetime

load_dotenv()

path = "Sample_Docs_Markdown/"
loader = DirectoryLoader(
    path,
    glob=["**/*.md", "**/*.mdx"],
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)

documents = loader.load()
documents = documents[:10]
print(f"Loaded documents: {len(documents)}")

for document in documents:
    document.metadata["filename"] = document.metadata["source"]

generator_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.7)
)
openai_client = openai.OpenAI()
generator_embeddings = OpenAIEmbeddings(client=openai_client)

# Persona and transforms (参考実装に合わせて追加)
personas = [
    Persona(
        name="LangChain Developer",
        role_description=(
            "I am a developer learning LangChain and reading its docs. "
            "I ask practical questions about usage, APIs, and examples in English."
        ),
    )
]
transforms = [HeadlineSplitter(), NERExtractor()]

# Query distribution: シングルホップの具体的質問を中心に生成
distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

# より細かく制御するため、直接コンストラクタを使用
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
    persona_list=personas,
)

TARGET_QUESTIONS = 3
testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=TARGET_QUESTIONS,
    transforms=transforms,
    query_distribution=distribution,
    with_debugging_logs=True,
)

# 生成結果の簡易出力
try:
    eval_dataset = testset.to_evaluation_dataset()
    print("--- Generated test samples ---")
    for i, sample in enumerate(eval_dataset[:TARGET_QUESTIONS]):
        print(f"Q{i+1}: {sample.user_input}")
        print(f"Ref : {sample.reference}")
        print("-" * 30)
except Exception as e:
    print(f"Preview failed: {e}")
