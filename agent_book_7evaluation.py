from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.transforms.extractors import (
    EmbeddingExtractor,
    SummaryExtractor,
    HeadlinesExtractor,
)
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.engine import Parallel
from ragas.testset.graph import NodeType
from ragas.utils import num_tokens_from_string
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai
from datetime import datetime
from typing import Optional
import argparse

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

# Optional personas / transforms
# デフォルトは「指定しない（ライブラリのデフォルトに委ねる）」に変更
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--use-custom-persona", action="store_true")
parser.add_argument("--auto-persona", action="store_true", help="use Ragas auto persona generation")
parser.add_argument("--use-custom-transforms", action="store_true")
parser.add_argument("--use-default-transforms", action="store_true", help="use Ragas default transforms")
parser.add_argument(
    "--safe-default-transforms",
    action="store_true",
    help="use a resilient default-like transforms that skip nodes missing embeddings",
)
args, _ = parser.parse_known_args()

personas: Optional[list[Persona]] = [
    Persona(
        name="LangChain Developer",
        role_description=(
            "I am a developer learning LangChain and reading its docs. "
            "I ask practical questions about usage, APIs, and examples in English."
        ),
    )
]
transforms = [NERExtractor(llm=generator_llm)]

if args.use_custom_persona:
    # same as default; retained for backwards compatibility
    pass
elif args.auto_persona:
    personas = None  # let Ragas generate personas from KG (requires summary embeddings)

if args.use_custom_transforms:
    transforms = [HeadlineSplitter(), NERExtractor(llm=generator_llm)]
elif args.use_default_transforms:
    transforms = None  # delegate to Ragas default (may include cosine similarity)
elif args.safe_default_transforms:
    # Build a resilient variant of default_transforms that avoids crashes when
    # some documents fail to produce summary embeddings.
    def _filter_doc_tokens(node, min_tokens=500):
        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties["page_content"]) > min_tokens
        )

    def _filter_docs(node):
        return node.type == NodeType.DOCUMENT

    def _filter_chunks(node):
        return node.type == NodeType.CHUNK

    # Count token distribution similar to ragas default
    def _count_bins(docs, bin_ranges):
        data = [num_tokens_from_string(doc.page_content) for doc in docs]
        bins = {f"{start}-{end}": 0 for start, end in bin_ranges}
        for num in data:
            for start, end in bin_ranges:
                if start <= num <= end:
                    bins[f"{start}-{end}"] += 1
                    break
        return bins

    bin_ranges = [(0, 100), (101, 500), (501, float("inf"))]
    result = _count_bins(documents, bin_ranges)
    result = {k: v / len(documents) for k, v in result.items()}

    tf = []
    if result["501-inf"] >= 0.25:
        # Large docs path
        # We mirror default ordering and include safe guards
        headlines_extractor = HeadlinesExtractor(
            llm=generator_llm, filter_nodes=lambda n: _filter_doc_tokens(n)
        )
        headlines_splitter = HeadlineSplitter(min_tokens=500)
        summary_extractor = SummaryExtractor(
            llm=generator_llm, filter_nodes=lambda n: _filter_doc_tokens(n)
        )
        node_filter = CustomNodeFilter(
            llm=generator_llm, filter_nodes=lambda n: _filter_chunks(n)
        )
        summary_emb_extractor = EmbeddingExtractor(
            embedding_model=generator_embeddings,
            property_name="summary_embedding",
            embed_property_name="summary",
            filter_nodes=lambda n: _filter_doc_tokens(n),
        )
        theme_extractor = ThemesExtractor(
            llm=generator_llm, filter_nodes=lambda n: _filter_chunks(n)
        )
        ner_extractor = NERExtractor(
            llm=generator_llm, filter_nodes=lambda n: _filter_chunks(n)
        )
        # Safe cosine: only include docs that actually have summary_embedding
        cosine_sim_builder = CosineSimilarityBuilder(
            property_name="summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.7,
            filter_nodes=lambda n: _filter_doc_tokens(n)
            and n.properties.get("summary_embedding") is not None,
        )
        ner_overlap_sim = OverlapScoreBuilder(
            threshold=0.01, filter_nodes=lambda n: _filter_chunks(n)
        )
        transforms = [
            # Extract headlines and split
            headlines_extractor,
            headlines_splitter,
            # Summarize documents
            summary_extractor,
            # Filter some chunks
            node_filter,
            # Compute embeddings/themes/ner
            Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
            # Build relationships safely
            Parallel(cosine_sim_builder, ner_overlap_sim),
        ]
    elif result["101-500"] >= 0.25:
        summary_extractor = SummaryExtractor(
            llm=generator_llm, filter_nodes=lambda n: _filter_doc_tokens(n, 100)
        )
        summary_emb_extractor = EmbeddingExtractor(
            embedding_model=generator_embeddings,
            property_name="summary_embedding",
            embed_property_name="summary",
            filter_nodes=lambda n: _filter_doc_tokens(n, 100),
        )
        # Safe cosine: only include docs that actually have summary_embedding
        cosine_sim_builder = CosineSimilarityBuilder(
            property_name="summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.5,
            filter_nodes=lambda n: _filter_doc_tokens(n, 100)
            and n.properties.get("summary_embedding") is not None,
        )
        ner_extractor = NERExtractor(llm=generator_llm)
        ner_overlap_sim = OverlapScoreBuilder(threshold=0.01)
        theme_extractor = ThemesExtractor(llm=generator_llm, filter_nodes=lambda n: _filter_docs(n))
        node_filter = CustomNodeFilter(llm=generator_llm)
        transforms = [
            summary_extractor,
            node_filter,
            Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
            Parallel(cosine_sim_builder, ner_overlap_sim),
        ]
    else:
        # Too short docs: fall back to minimal safe transforms
        transforms = [NERExtractor(llm=generator_llm)]

# Query distribution: シングルホップの具体的質問を中心に生成
distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

# より細かく制御するため、直接コンストラクタを使用
# personas を未指定(None)の場合はキーワード引数を渡さずデフォルト挙動に委ねる
generator_kwargs = dict(llm=generator_llm, embedding_model=generator_embeddings)
if personas is not None:
    generator_kwargs["persona_list"] = personas

generator = TestsetGenerator(**generator_kwargs)

TARGET_QUESTIONS = 3
gen_kwargs = dict(
    testset_size=TARGET_QUESTIONS,
    query_distribution=distribution,
    with_debugging_logs=True,
)
if transforms is not None:
    gen_kwargs["transforms"] = transforms

try:
    testset = generator.generate_with_langchain_docs(documents, **gen_kwargs)
except ValueError as e:
    msg = str(e)
    if args.auto_persona and "No nodes that satisfied the given filer" in msg:
        # Fallback: use a default persona when auto-generation can't find eligible nodes
        fallback_personas = [
            Persona(
                name="LangChain Developer",
                role_description=(
                    "I am a developer learning LangChain and reading its docs. "
                    "I ask practical questions about usage, APIs, and examples in English."
                ),
            )
        ]
        generator_kwargs["persona_list"] = fallback_personas
        generator = TestsetGenerator(**generator_kwargs)
        testset = generator.generate_with_langchain_docs(documents, **gen_kwargs)
    else:
        raise

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
