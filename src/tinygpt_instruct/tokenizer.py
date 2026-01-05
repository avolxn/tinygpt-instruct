import os
from collections.abc import Iterator

from tokenizers import Regex as HFRegex
from tokenizers import Tokenizer as HFTokenizerBase
from tokenizers import decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|system_start|>",
    "<|system_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


class Tokenizer:
    """Быстрый BPE токенизатор на основе HuggingFace Tokenizers."""

    def __init__(self):
        """Инициализирует пустой токенизатор."""
        self._tokenizer: HFTokenizerBase | None = None
        self.special_tokens: dict[str, int] = {}

    def train(
        self,
        text_iterator: Iterator[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
    ) -> None:
        """Обучает токенизатор на итераторе текстов.

        Args:
            text_iterator: Итератор строк для обучения.
            vocab_size: Целевой размер словаря.
            special_tokens: Список специальных токенов (по умолчанию SPECIAL_TOKENS).
        """
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS

        tokenizer = HFTokenizerBase(
            BPE(
                byte_fallback=True,
                unk_token=None,
                fuse_unk=False,
            )
        )
        tokenizer.normalizer = None
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=HFRegex(GPT4_SPLIT_PATTERN),
                    behavior="isolated",
                    invert=False,
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special_tokens,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)

        self._tokenizer = tokenizer
        self.special_tokens = {
            token.content: token_id for token_id, token in tokenizer.get_added_tokens_decoder().items()
        }

    def encode(self, text: str) -> list[int]:
        """Кодирует текст в список ID токенов.

        Args:
            text: Входной текст.

        Returns:
            Список ID токенов.
        """
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        """Декодирует список ID токенов обратно в строку.

        Args:
            ids: Список ID токенов.

        Returns:
            Декодированная строка.
        """
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, model_folder_path: str) -> None:
        """Сохраняет токенизатор в указанную папку.

        Args:
            model_folder_path: Путь к папке для сохранения.
        """
        os.makedirs(model_folder_path, exist_ok=True)
        tokenizer_path = os.path.join(model_folder_path, "tokenizer.json")
        self._tokenizer.save(tokenizer_path)

    @classmethod
    def load(cls, model_path: str) -> "Tokenizer":
        """Загружает токенизатор из файла.

        Args:
            model_path: Путь к файлу tokenizer.json или папке с ним.

        Returns:
            Загруженный экземпляр Tokenizer.
        """
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, "tokenizer.json")
        instance = cls()
        instance._tokenizer = HFTokenizerBase.from_file(model_path)
        instance.special_tokens = {
            token.content: token_id for token_id, token in instance._tokenizer.get_added_tokens_decoder().items()
        }
        return instance

    def render_conversation(
        self,
        conversation: dict,
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """Токенизирует диалог для instruction-tuning.

        Args:
            conversation: Словарь с ключом "messages".
            max_tokens: Максимальное количество токенов.

        Returns:
            Кортеж (ids, mask), где mask=1 для токенов assistant.
        """
        ids: list[int] = []
        mask: list[int] = []

        messages = conversation.get("messages", [])
        if not messages:
            return ids, mask

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            nonlocal ids, mask
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        add_tokens(self.special_tokens["<|bos|>"], 0)

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                add_tokens(self.special_tokens["<|system_start|>"], 0)
                add_tokens(self.encode(content), 0)
                add_tokens(self.special_tokens["<|system_end|>"], 0)

            elif role == "user":
                add_tokens(self.special_tokens["<|user_start|>"], 0)
                add_tokens(self.encode(content), 0)
                add_tokens(self.special_tokens["<|user_end|>"], 0)

            elif role == "assistant":
                add_tokens(self.special_tokens["<|assistant_start|>"], 0)

                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(self.special_tokens["<|python_start|>"], 1)
                            add_tokens(value_ids, 1)
                            add_tokens(self.special_tokens["<|python_end|>"], 1)
                        elif part["type"] == "python_output":
                            add_tokens(self.special_tokens["<|output_start|>"], 0)
                            add_tokens(value_ids, 0)
                            add_tokens(self.special_tokens["<|output_end|>"], 0)

                add_tokens(self.special_tokens["<|assistant_end|>"], 1)

        return ids[:max_tokens], mask[:max_tokens]
