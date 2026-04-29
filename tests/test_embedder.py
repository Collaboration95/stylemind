from __future__ import annotations

import pytest


# Unit test: LocalEmbedder returns correct dimensions (384)
@pytest.mark.unit
def test_local_embedder_dimensions():
    from stylemind.rag.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    assert embedder.dimensions == 384


# Unit test: LocalEmbedder.embed_query returns list[float] of length 384
@pytest.mark.unit
def test_local_embedder_embed_query():
    from stylemind.rag.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    result = embedder.embed_query("blue summer dress")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


# Unit test: LocalEmbedder.embed_batch returns list of correct shape
@pytest.mark.unit
def test_local_embedder_embed_batch():
    from stylemind.rag.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    texts = ["casual summer dress", "office blazer", "leather boots"]
    results = embedder.embed_batch(texts)
    assert len(results) == 3
    for r in results:
        assert len(r) == 384


# Unit test: get_embedder returns LocalEmbedder by default
@pytest.mark.unit
def test_get_embedder_local(mock_env):
    from stylemind.config import EmbeddingConfig
    from stylemind.rag.embedder import LocalEmbedder, get_embedder

    config = EmbeddingConfig(provider="local", model_name="sentence-transformers/all-MiniLM-L6-v2", dimensions=384)
    embedder = get_embedder(config)
    assert isinstance(embedder, LocalEmbedder)


# Unit test: build_product_text produces deterministic consistent text
@pytest.mark.unit
def test_build_product_text_deterministic():
    from scripts.embed import build_product_text

    record = {
        "name": "Linen Trouser",
        "description": "Wide-leg linen trouser",
        "aesthetics": ["Coastal Grandma"],
        "occasions": ["Casual", "Weekend Brunch"],
        "color_palette": "Earthy Neutrals",
    }
    text1 = build_product_text(record)
    text2 = build_product_text(record)
    assert text1 == text2
    assert "Linen Trouser" in text1
    assert "Coastal Grandma" in text1
    assert "Earthy Neutrals" in text1


# Unit test: build_aesthetic_text includes name, description, keywords
@pytest.mark.unit
def test_build_aesthetic_text():
    from scripts.embed import build_aesthetic_text

    record = {
        "name": "Quiet Luxury",
        "description": "Understated sophistication",
        "keywords": ["minimal", "refined"],
    }
    text = build_aesthetic_text(record)
    assert "Quiet Luxury" in text
    assert "Understated sophistication" in text
    assert "minimal" in text
    assert "refined" in text


# Unit test: OpenAIEmbedder dimensions property returns configured value
@pytest.mark.unit
def test_openai_embedder_dimensions():
    from unittest.mock import MagicMock, patch

    with patch("openai.OpenAI", return_value=MagicMock()):
        from stylemind.rag.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder(api_key="test", model="text-embedding-3-small", dimensions=1536)
        assert embedder.dimensions == 1536
