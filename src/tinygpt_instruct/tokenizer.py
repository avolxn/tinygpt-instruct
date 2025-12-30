import os

import regex as re

GPT2_SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


class Tokenizer:
    """Byte-level BPE tokenizer with regex pre-tokenization"""

    def __init__(self, split_pattern: str | None = None):
        """Инициализирует токенизатор.

        Args:
            split_pattern (str | None): Шаблон регулярного выражения для предварительного разбиения текста.
        """
        self.split_pattern: re.Pattern = re.compile(split_pattern or ".+")
        self.special_tokens: dict[str, int] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}

    def _build_vocab(self) -> dict[int, bytes]:
        """Собирает словарь токенов на основе выполненных объединений и спецтокенов.

        Returns:
            dict[int, bytes]: Словарь, где ключ — ID токена, значение — его байты.
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for (pair0, pair1), idx in self.merges.items():
            vocab[idx] = vocab[pair0] + vocab[pair1]

        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode("utf-8")

        return vocab

    def _merge(self, ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """Заменяет все вхождения пары токенов на новый токен.

        Args:
            ids (list[int]): Список ID токенов.
            pair (tuple[int, int]): Пара токенов для объединения.
            idx (int): Новый ID токена.

        Returns:
            list[int]: Новый список ID после объединения.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _get_stats(
        self,
        ids: list[int],
        stats: dict[tuple[int, int], int] | None = None,
    ) -> dict[tuple[int, int], int]:
        """Подсчитывает частоту встречаемости пар последовательных токенов.

        Args:
            ids (list[int]): Список ID токенов.
            stats (dict[tuple[int, int], int] | None): Существующий словарь частот для обновления.

        Returns:
            dict[tuple[int, int], int]: Обновленный словарь частот пар токенов.
        """
        stats = stats if stats is not None else {}

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            stats[pair] = stats.get(pair, 0) + 1

        return stats

    def register_special_tokens(self, special_tokens: list[str]) -> None:
        """Регистрирует новые специальные токены и назначает им ID.

        Args:
            special_tokens (list[str]): Список строк для регистрации в качестве спецтокенов.
        """
        idx = len(self.vocab)
        for special_token in special_tokens:
            if special_token not in self.special_tokens:
                self.special_tokens[special_token] = idx
                idx += 1

        self.vocab = self._build_vocab()

    def train(self, text: str, vocab_size: int) -> None:
        """Обучает токенизатор на заданном тексте до достижения указанного размера словаря.

        Args:
            text (str): Текст для обучения.
            vocab_size (int): Целевой размер словаря (минимум 256).

        Raises:
            ValueError: Если vocab_size меньше 256.
        """
        num_merges = vocab_size - 256
        if num_merges < 0:
            raise ValueError("Размер словаря должен быть не менее 256")

        chunks = re.findall(self.split_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        merges = {}

        for _ in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                stats = self._get_stats(chunk_ids, stats)

            if not stats:
                break

            best_pair = max(stats, key=lambda pair: stats[pair])
            idx = 256 + len(merges)
            ids = [self._merge(chunk_ids, best_pair, idx) for chunk_ids in ids]

            merges[best_pair] = idx

        self.merges = merges
        self.vocab = self._build_vocab()

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Кодирует отдельный чанк текста (без спецтокенов) в список ID.

        Args:
            chunk (str): Фрагмент текста.

        Returns:
            list[int]: Список ID токенов.
        """
        ids = list(chunk.encode("utf-8"))

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)

        return ids

    def encode(self, text: str) -> list[int]:
        """Кодирует текст в список ID токенов, включая обработку специальных токенов.

        Args:
            text (str): Входной текст.

        Returns:
            list[int]: Список ID токенов.
        """
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
            parts = re.split(special_pattern, text)
        else:
            parts = [text]

        ids = []
        for part in parts:
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                chunks = re.findall(self.split_pattern, part)
                for chunk in chunks:
                    ids.extend(self._encode_chunk(chunk))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Декодирует список ID токенов обратно в строку.

        Args:
            ids (list[int]): Список ID токенов.

        Returns:
            str: Декодированная строка (с заменой некорректных байтов).
        """
        tokens = [self.vocab[idx] for idx in ids]
        full_bytes = b"".join(tokens)

        return full_bytes.decode("utf-8", errors="replace")

    def save(self, model_folder_path: str) -> None:
        """Сохраняет модель токенизатора в указанную папку.

        Args:
            model_folder_path (str): Путь к папке, где будет создан файл tokenizer.model.
        """
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path = os.path.join(model_folder_path, "tokenizer.model")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(f"{self.split_pattern.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            f.write(f"{len(self.merges)}\n")

            for special_token, idx in self.special_tokens.items():
                f.write(f"{special_token} {idx}\n")

            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")

    def load(self, model_path: str) -> None:
        """Загружает модель токенизатора из файла формата .model.

        Args:
            model_path (str): Полный путь к файлу модели.

        Raises:
            ValueError: Если путь не ведет к файлу формата .model.
        """
        if not model_path.endswith(".model"):
            raise ValueError("Путь к модели должен вести к файлу формата .model")

        with open(model_path, encoding="utf-8") as f:
            split_pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            num_merges = int(f.readline().strip())

            special_tokens = {}
            for _ in range(num_special):
                token, idx = f.readline().strip().split()
                special_tokens[token] = int(idx)

            merges = {}
            for _ in range(num_merges):
                p0, p1, idx = map(int, f.readline().strip().split())
                merges[(p0, p1)] = idx

            self.split_pattern = re.compile(split_pattern)
            self.special_tokens = special_tokens
            self.merges = merges
            self.vocab = self._build_vocab()
