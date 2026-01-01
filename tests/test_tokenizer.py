import os

import pytest

from tinygpt_instruct.tokenizer import GPT4_SPLIT_PATTERN, Tokenizer, HuggingFaceTokenizer

TEST_STRINGS = [
    "",
    "?",
    "hello world!!!?",
    "привет мир!",
    "lol123 😉",
]


@pytest.fixture
def trained_tokenizer():
    text_data = [" ".join(TEST_STRINGS) + " hello world"]
    tokenizer = Tokenizer()
    tokenizer.train(iter(text_data), 256 + 10 + 11)  # +11 for special tokens
    return tokenizer


@pytest.fixture
def trained_hf_tokenizer():
    text_data = [" ".join(TEST_STRINGS) + " hello world"]
    tokenizer = HuggingFaceTokenizer()
    tokenizer.train(iter(text_data), 256 + 10 + 11)
    return tokenizer


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_encode_decode_identity(trained_tokenizer: Tokenizer, text: str):
    encoded_ids = trained_tokenizer.encode(text)
    decoded_text = trained_tokenizer.decode(encoded_ids)
    assert decoded_text == text


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_hf_encode_decode_identity(trained_hf_tokenizer: HuggingFaceTokenizer, text: str):
    encoded_ids = trained_hf_tokenizer.encode(text)
    decoded_text = trained_hf_tokenizer.decode(encoded_ids)
    assert decoded_text == text


def test_wikipedia_example():
    text = "aaabdaaabac"
    tokenizer = Tokenizer()
    tokenizer.train(iter([text]), 256 + 3 + 11)

    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_special_tokens():
    tokenizer = Tokenizer()
    tokenizer.train(iter(["hello world"]), 256 + 5 + 11)

    text = "hello <|bos|> world"
    ids = tokenizer.encode(text)

    assert tokenizer.special_tokens["<|bos|>"] in ids
    assert tokenizer.decode(ids) == text


def test_save_load(tmp_path):
    test_dir = os.path.dirname(__file__)
    file_path = os.path.join(test_dir, "test_tokenizer_data.txt")
    with open(file_path, encoding="utf-8") as f:
        test_text = f.read()

    tokenizer = Tokenizer()
    tokenizer.train(iter([test_text]), 256 + 32 + 11)

    original_text = "Testing save/load with <|bos|> token."
    original_ids = tokenizer.encode(original_text)

    tokenizer.save(str(tmp_path))

    # Проверяем что файл называется tokenizer_bpe.json
    assert (tmp_path / "tokenizer_bpe.json").exists()

    new_tokenizer = Tokenizer.load(str(tmp_path))

    assert new_tokenizer.encode(original_text) == original_ids
    assert new_tokenizer.decode(original_ids) == original_text
    assert new_tokenizer.split_pattern.pattern == tokenizer.split_pattern.pattern


def test_hf_save_load(tmp_path):
    test_dir = os.path.dirname(__file__)
    file_path = os.path.join(test_dir, "test_tokenizer_data.txt")
    with open(file_path, encoding="utf-8") as f:
        test_text = f.read()

    tokenizer = HuggingFaceTokenizer()
    tokenizer.train(iter([test_text]), 256 + 32 + 11)

    original_text = "Testing save/load with <|bos|> token."
    original_ids = tokenizer.encode(original_text)

    tokenizer.save(str(tmp_path))

    new_tokenizer = HuggingFaceTokenizer.load(str(tmp_path))

    assert new_tokenizer.encode(original_text) == original_ids
    assert new_tokenizer.decode(original_ids) == original_text
