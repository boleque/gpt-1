import sys
import os
import time

# Add src directory to path so we can import BPE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpe import BPE


def test_bpe_example_1():
    """Test BPE with example 1."""
    vocab_size = 30
    text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'

    expected_tokens = [
        ' ', '.', 'В', 'И', 'а', 'б', 'в', 'г', 'е', 'з', 'и', 'к', 'л', 'о', 'п', 'р',
        'с', 'т', 'у', 'ш', 'я', 'уз', 'узо', 'узов', 'а ', 'гр', ' к', ' кузов', ' гр', 'а а'
    ]

    bpe = BPE(vocab_size)

    # Measure execution time
    start_time = time.time()
    result = bpe.fit(text)
    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Vocab size: {vocab_size}")
    print(f"Text length: {len(text)} characters")
    print(f"Text: {text}")
    print(f"Expected tokens: {expected_tokens}")
    print(f"Result tokens: {result}")
    print(f"Expected count: {len(expected_tokens)}")
    print(f"Result count: {len(result) if result else 0}")
    print(f"⏱️  Execution time: {execution_time:.4f} seconds ({execution_time * 1000:.2f} ms)")

    assert result is not None, "fit() should return tokens"
    assert len(result) == vocab_size, f"Expected {vocab_size} tokens, got {len(result)}"
    assert set(result) == set(expected_tokens), "Token sets should match"

    print("✓ Test example 1 passed!")
    return execution_time


def test_bpe_example_2():
    """Test BPE with example 2."""
    vocab_size = 31
    text = 'Однажды был случай в далёком Макао: макака коалу в какао макала, коала лениво какао лакала, макака макала, коала икала.'

    expected_tokens = [
        ' ', ',', '.', ':', 'М', 'О', 'а', 'б', 'в', 'д', 'е', 'ж', 'и', 'й', 'к', 'л',
        'м', 'н', 'о', 'с', 'у', 'ч', 'ы', 'ё', 'ка', 'ла', 'ака', 'ко', ' м', ' мака', ' ко'
    ]

    bpe = BPE(vocab_size)

    # Measure execution time
    start_time = time.time()
    result = bpe.fit(text)
    end_time = time.time()

    execution_time = end_time - start_time

    print(f"\nVocab size: {vocab_size}")
    print(f"Text length: {len(text)} characters")
    print(f"Text: {text}")
    print(f"Expected tokens: {expected_tokens}")
    print(f"Result tokens: {result}")
    print(f"Expected count: {len(expected_tokens)}")
    print(f"Result count: {len(result) if result else 0}")
    print(f"⏱️  Execution time: {execution_time:.4f} seconds ({execution_time * 1000:.2f} ms)")

    assert result is not None, "fit() should return tokens"
    assert len(result) == vocab_size, f"Expected {vocab_size} tokens, got {len(result)}"
    assert set(result) == set(expected_tokens), "Token sets should match"

    print("✓ Test example 2 passed!")
    return execution_time


if __name__ == "__main__":
    print("Running BPE tests...\n")
    print("=" * 70)

    total_time = 0.0
    test_count = 0

    try:
        time1 = test_bpe_example_1()
        total_time += time1
        test_count += 1
    except AssertionError as e:
        print(f"✗ Test example 1 failed: {e}")
    except Exception as e:
        print(f"✗ Test example 1 error: {e}")

    try:
        time2 = test_bpe_example_2()
        total_time += time2
        test_count += 1
    except AssertionError as e:
        print(f"✗ Test example 2 failed: {e}")
    except Exception as e:
        print(f"✗ Test example 2 error: {e}")

    print("\n" + "=" * 70)
    print(f"All tests completed!")
    if test_count > 0:
        print(f"\n📊 Performance Summary:")
        print(f"   Total execution time: {total_time:.4f} seconds ({total_time * 1000:.2f} ms)")
        print(f"   Average per test: {total_time / test_count:.4f} seconds ({(total_time / test_count) * 1000:.2f} ms)")
    print("=" * 70)
