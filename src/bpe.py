from collections import OrderedDict, deque
import heapq

class BPE:
    vocab_size: int

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def fit(self, text):
        tokens: list = self.__retrieve_tokens(text)
        self.__id2token(tokens)
        self.__token2id(tokens)

        return tokens

    def __retrieve_tokens(self, text: str):
        unique_tokens_set = set(text)
        unique_tokens = sorted(unique_tokens_set)

        size_diff = len(unique_tokens) - self.vocab_size
        if size_diff > 0:
            return unique_tokens[:-size_diff]

        individual_chars = [ch for ch in text]
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
            individual_chars = self.__merge_pair(individual_chars, most_frequent_pair)
            unique_tokens_set.add(most_frequent_pair)
            unique_tokens.append(most_frequent_pair)
        return unique_tokens

    def __merge_pair(self, tokens, pair):
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
        pass

    def __token2id(self, tokens):
        pass

if __name__ == "__main__":
    vocab_size = 30
    text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'
    bpe = BPE(vocab_size)
    bpe.fit(text)
