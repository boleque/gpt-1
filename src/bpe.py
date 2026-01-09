from collections import OrderedDict, deque
import heapq

class BPE:
    vocab_size: int

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def fit(self, text):
        # Get unique characters and sort them
        unique_tokens_set = set(text)
        unique_tokens = sorted(unique_tokens_set)

        size_diff = len(unique_tokens) - self.vocab_size
        if size_diff > 0:
            unique_tokens = unique_tokens[:-size_diff]
        else:
            self.__study(unique_tokens)

        # TODO handle edge case unique_tokens_set is already bigger than vocab size

        individual_chars = [ch for ch in text]
        while True:
            if len(unique_tokens) >= self.vocab_size:
                break;

            pair_frequencies = OrderedDict()
            for idx in range(len(individual_chars) - 1):

                pair = ''.join(individual_chars[idx: idx + 2])
                if pair in unique_tokens_set:
                    continue

                if pair in pair_frequencies:
                    value = pair_frequencies[pair]
                    value[0] += 1
                    value.append(idx)
                    pair_frequencies[pair] = value
                else:
                    pair_frequencies[pair] = [1, idx]

            max_frequency = 0
            most_frequent_pair_info = None

            for pair, frequency in pair_frequencies.items():
                if frequency[0] > max_frequency:
                   max_frequency = frequency[0]
                   most_frequent_pair_info = (pair, frequency[1:])

            token, indexes = most_frequent_pair_info

            unique_tokens.append(token)
            individual_chars = self._replace_item(individual_chars, indexes)
            unique_tokens_set.add(token)

        return unique_tokens

    def __study(self, text: str):
        pass

    def __id2token(self, tokens):
        pass

    def __token2id(self, tokens):
        pass

    def _replace_item(self, individual_chars, indexes):
        new_individual_chars = []
        token = None
        for idx, value in enumerate(individual_chars):
            if idx in indexes:
                token = value
                continue
            elif token != None:
                token += value
                new_individual_chars.append(token)
                token = None
            else:
                new_individual_chars.append(value)

        return new_individual_chars


if __name__ == "__main__":
    vocab_size = 30
    text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'
    bpe = BPE(vocab_size)
    bpe.fit(text)
