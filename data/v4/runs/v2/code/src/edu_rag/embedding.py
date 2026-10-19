"""Lazy, local BGE-M3 inference shared by indexing and queries."""

from functools import lru_cache
from pathlib import Path
from threading import Lock

from edu_rag.config import ConfigurationError, VECTOR_DIMENSIONS, get_settings


class EmbeddingError(RuntimeError):
    pass


def local_snapshot():
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(get_settings().embedding_model, "config.json")
    if not isinstance(config, str) or not Path(config).is_file():
        raise ConfigurationError("本机尚未缓存 BGE-M3，请先完成模型下载。")
    return Path(config).parent


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(local_snapshot()), local_files_only=True)


def count_tokens(text):
    return len(get_tokenizer().encode(text, add_special_tokens=True, truncation=False))


class BGEEncoder:
    def __init__(self):
        settings = get_settings()
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ConfigurationError("请先安装 requirements-rag-lock.txt 中的模型依赖。") from None
        snapshot = local_snapshot()
        self.model_id = embedding_model_id()
        self.max_length = settings.embedding_max_length
        self.lock = Lock()
        try:
            self.model = BGEM3FlagModel(
                str(snapshot), use_fp16=False, normalize_embeddings=True,
                devices=settings.embedding_device,
            )
        except Exception:
            raise EmbeddingError("BGE-M3 加载失败，请检查模型文件和可用内存。") from None

    def encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        if not texts:
            return []
        if any(count_tokens(value) > self.max_length for value in texts):
            raise EmbeddingError("文本超过向量模型的 token 上限，请先切分正文或缩短问题；未截断文本。")
        try:
            with self.lock:
                result = self.model.encode(
                    texts, batch_size=4, max_length=self.max_length,
                    return_dense=True, return_sparse=False, return_colbert_vecs=False,
                )["dense_vecs"]
        except Exception:
            raise EmbeddingError("向量生成失败，请检查模型运行环境和可用内存。") from None
        values = np.asarray(result)
        if (values.shape != (len(texts), VECTOR_DIMENSIONS)
                or not np.isfinite(values).all()
                or (np.linalg.norm(values, axis=1) == 0).any()):
            raise EmbeddingError("模型输出不是有效的 1024 维向量。")
        return values.tolist()


@lru_cache(maxsize=1)
def get_encoder() -> BGEEncoder:
    return BGEEncoder()


def embedding_model_id():
    """Resolve the index fingerprint without loading neural model weights."""
    settings = get_settings()
    return (f"{settings.embedding_model}@{local_snapshot().name}"
            f":dense-l2-fp32:maxlen={settings.embedding_max_length}")
