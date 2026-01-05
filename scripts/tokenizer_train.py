import argparse
import time

import torch

from tinygpt_instruct.tokenizer import Tokenizer
from tinygpt_instruct.utils import get_models_dir

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument(
    "--max_chars",
    type=int,
    default=10_000_000_000,
    help="Maximum characters to train on (default: 10B)",
)
parser.add_argument(
    "--doc_cap",
    type=int,
    default=10_000,
    help="Maximum characters per document (default: 10,000)",
)
parser.add_argument(
    "--vocab_size",
    type=int,
    default=65536,
    help="Vocabulary size (default: 65536 = 2^16)",
)
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")


def text_iterator():
    # TODO
    pass


# Обучение
tokenizer = Tokenizer()
train_start = time.time()
tokenizer.train(
    text_iterator(),
    vocab_size=args.vocab_size,
)
train_end = time.time()
train_time = train_end - train_start
print(f"Training time: {train_time:.2f}s")

# Сохранение токенизатора
models_dir = get_models_dir()
tokenizer_dir = models_dir / "tokenizer"
tokenizer.save(str(tokenizer_dir))
print(f"Saved tokenizer to {tokenizer_dir}")

# Тестирование
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""

encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
if not decoded == test_text:
    print(f"Test text:\n{test_text}\n")
    print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
    print(f"Decoded:\n{decoded}\n")
    raise ValueError("Encode/decode test failed!")
else:
    print("Encode/decode test passed")

# Сохранение token_bytes
token_bytes = tokenizer.get_token_bytes()
token_bytes_path = tokenizer_dir / "token_bytes.pt"
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Логирование статистики
special_set = set(tokenizer.get_special_tokens())
token_bytes_nonzero = token_bytes[token_bytes > 0].to(dtype=torch.float32)
print("\nToken bytes statistics:")
print(f"- Number of special tokens: {len(special_set)}")
print(f"- Min bytes per token: {int(token_bytes_nonzero.min().item())}")
print(f"- Max bytes per token: {int(token_bytes_nonzero.max().item())}")
print(f"- Mean bytes per token: {token_bytes_nonzero.mean().item():.2f}")
print(f"- Std bytes per token: {token_bytes_nonzero.std().item():.2f}")
