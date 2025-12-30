import os

import pytest

from tinygpt_instruct.tokenizer import GPT2_SPLIT_PATTERN, Tokenizer

TEST_STRINGS = [
    "",
    "?",
    "hello world!!!?",
    "привет мир!",
    "lol123 😉",
]


@pytest.fixture
def trained_tokenizer():
    tokenizer = Tokenizer(GPT2_SPLIT_PATTERN)
    tokenizer.train(" ".join(TEST_STRINGS) + " hello world", 256 + 10)
    return tokenizer


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_encode_decode_identity(trained_tokenizer: Tokenizer, text: str):
    encoded_ids = trained_tokenizer.encode(text)
    decoded_text = trained_tokenizer.decode(encoded_ids)
    assert decoded_text == text


def test_wikipedia_example():
    tokenizer = Tokenizer(".+")
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3)

    ids = tokenizer.encode(text)
    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(ids) == text


def test_special_tokens():
    tokenizer = Tokenizer(GPT2_SPLIT_PATTERN)
    tokenizer.register_special_tokens(["<|endoftext|>"])
    tokenizer.train("hello world", 256 + 5)

    text = "hello <|endoftext|> world"
    ids = tokenizer.encode(text)

    assert tokenizer.special_tokens["<|endoftext|>"] in ids
    assert tokenizer.decode(ids) == text


def test_save_load(tmp_path):
    test_dir = os.path.dirname(__file__)
    file_path = os.path.join(test_dir, "test_tokenizer_data.txt")
    with open(file_path, encoding="utf-8") as f:
        test_text = f.read()

    tokenizer = Tokenizer(GPT2_SPLIT_PATTERN)
    tokenizer.train(test_text, 256 + 32)
    tokenizer.register_special_tokens(["<|extra|>"])

    original_text = "Testing save/load with <|extra|> token."
    original_ids = tokenizer.encode(original_text)

    tokenizer.save(str(tmp_path))

    new_tokenizer = Tokenizer(".+")
    new_tokenizer.load(str(tmp_path / "tokenizer.model"))

    assert new_tokenizer.encode(original_text) == original_ids
    assert new_tokenizer.decode(original_ids) == original_text
    assert new_tokenizer.split_pattern.pattern == tokenizer.split_pattern.pattern
