from collections import OrderedDict, deque
import heapq

class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size: int = vocab_size
        self.id2token: Dict[str, int] = {}
        self.token2id: Dict[str, int] = {}

    def fit(self, text):
        tokens: list = self.__generate_tokens(text)
        self.__id2token(tokens)
        self.__token2id(tokens)

        return tokens

    def encode(self, text: str):
        result = []
        i = 0

        while i < len(text):
            char = text[i]

            candidate_tokens = [token for token in self.token2id.keys() if token.startswith(char)]

            best_token = char
            for token in candidate_tokens:
                if i + len(token) <= len(text) and text[i: i + len(token)] == token:
                    if len(token) > len(best_token):
                        best_token = token

            result.append(best_token)

            i += len(best_token)

        return [self.token2id[token] for token in result]

    def __generate_tokens(self, text: str):
        unique_tokens_set = set(text)
        unique_tokens = sorted(unique_tokens_set)

        size_diff = len(unique_tokens) - self.vocab_size
        if size_diff > 0:
            return unique_tokens[:-size_diff]

        individual_chars = list(text)
        while True:
            if len(unique_tokens) >= self.vocab_size:
                break

            pair_frequencies = OrderedDict()
            for idx in range(len(individual_chars) - 1):
                pair = individual_chars[idx] + individual_chars[idx + 1]
                if pair in unique_tokens_set:
                    continue
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1

            most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
            individual_chars = BPE.__merge_pair(individual_chars, most_frequent_pair)
            unique_tokens_set.add(most_frequent_pair)
            unique_tokens.append(most_frequent_pair)
        return unique_tokens

    @staticmethod
    def __merge_pair(tokens, pair):
        result = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i] + tokens[i + 1]) == pair:
                result.append(pair)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def __id2token(self, tokens):
        self.id2token = {i: item for i, item in enumerate(tokens)}

    def __token2id(self, tokens):
        self.token2id = {item: i for i, item in enumerate(tokens)}

if __name__ == "__main__":
    vocab_size = 15
    text = 'вором дрова, дрова вширь двора'
    bpe = BPE(vocab_size)
    bpe.fit(text)
    result = bpe.encode(text)
    print(result)
