import sys
import os

# Add src directory to path so we can import BPE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bpe import BPE


def test_encode_basic():
    """Test basic encoding."""
    vocab_size = 30
    train_text = 'Ежик в тумане!'

    bpe = BPE(vocab_size)
    bpe.fit(train_text)

    # Encode text
    test_text = 'кузов'
    encoded = bpe.encode(test_text)

    print(f"Test text: '{test_text}'")
    print(f"Encoded: {encoded}")

    # Check result
    assert encoded is not None
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    print("✓ Test passed!")


def test_encode_single_char():
    """Test encoding single character."""
    vocab_size = 30
    train_text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'

    bpe = BPE(vocab_size)
    bpe.fit(train_text)

    # Encode single char
    test_text = 'а'
    encoded = bpe.encode(test_text)

    print(f"\nTest text: '{test_text}'")
    print(f"Encoded: {encoded}")

    assert encoded is not None
    assert len(encoded) == 1

    print("✓ Test passed!")


def test_encode_with_merged():
    """Test encoding with merged tokens."""
    vocab_size = 30
    train_text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'

    bpe = BPE(vocab_size)
    bpe.fit(train_text)

    # Encode text that should have merged tokens
    test_text = 'арбузов'
    encoded = bpe.encode(test_text)

    print(f"\nTest text: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Original length: {len(test_text)}, Encoded length: {len(encoded)}")

    assert encoded is not None
    assert len(encoded) > 0

    print("✓ Test passed!")


if __name__ == "__main__":
    print("Running encode tests...\n")

    try:
        test_encode_basic()
    except Exception as e:
        print(f"✗ Test failed: {e}")

    try:
        test_encode_single_char()
    except Exception as e:
        print(f"✗ Test failed: {e}")

    try:
        test_encode_with_merged()
    except Exception as e:
        print(f"✗ Test failed: {e}")

    print("\n✓ All tests completed!")
