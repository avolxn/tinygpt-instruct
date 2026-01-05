import os

import pytest

from tinygpt_instruct.tokenizer import Tokenizer

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


@pytest.mark.parametrize("text", TEST_STRINGS)
def test_encode_decode_identity(trained_tokenizer: Tokenizer, text: str):
    encoded_ids = trained_tokenizer.encode(text)
    decoded_text = trained_tokenizer.decode(encoded_ids)
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

    # Проверяем что файл называется tokenizer.json
    assert (tmp_path / "tokenizer.json").exists()

    new_tokenizer = Tokenizer.load(str(tmp_path))

    assert new_tokenizer.encode(original_text) == original_ids
    assert new_tokenizer.decode(original_ids) == original_text


def test_render_conversation(trained_tokenizer: Tokenizer):
    conversation = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    }

    ids, mask = trained_tokenizer.render_conversation(conversation)

    assert len(ids) == len(mask)
    assert ids[0] == trained_tokenizer.special_tokens["<|bos|>"]
    assert trained_tokenizer.special_tokens["<|system_start|>"] in ids
    assert trained_tokenizer.special_tokens["<|user_start|>"] in ids
    assert trained_tokenizer.special_tokens["<|assistant_start|>"] in ids

    # Check that mask is 1 only for assistant messages
    # system message: <|bos|> (0), <|system_start|> (0), content (0), <|system_end|> (0)
    # user message: <|user_start|> (0), content (0), <|user_end|> (0)
    # assistant message: <|assistant_start|> (0), content (1), <|assistant_end|> (1)

    # In current implementation:
    # <|bos|> -> 0
    # <|system_start|> -> 0
    # content -> 0
    # <|system_end|> -> 0
    # <|user_start|> -> 0
    # content -> 0
    # <|user_end|> -> 0
    # <|assistant_start|> -> 0
    # content -> 1
    # <|assistant_end|> -> 1

    # Find assistant content indices
    assistant_start_idx = ids.index(trained_tokenizer.special_tokens["<|assistant_start|>"])
    assistant_end_idx = ids.index(trained_tokenizer.special_tokens["<|assistant_end|>"])

    assert all(m == 1 for m in mask[assistant_start_idx + 1 : assistant_end_idx + 1])
    assert all(m == 0 for m in mask[: assistant_start_idx + 1])
