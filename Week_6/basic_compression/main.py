import sys
import pickle

def compress_text(text):
    word_to_id = {}
    compressed_text = []
    unique_id = 0

    for word in text.split():
        if word not in word_to_id:
            word_to_id[word] = unique_id
            unique_id += 1
        compressed_text.append(word_to_id[word])

    return compressed_text, word_to_id

def decompress_text(compressed_text, word_to_id):
    id_to_word = {id: word for word, id in word_to_id.items()}
    return ' '.join(id_to_word[id] for id in compressed_text)

if __name__ == "__main__":
    # Original text
    text = "this is a test this is only a test"
    original_size = sys.getsizeof(text)
    print(f"Original size: {original_size} bytes")

    # Compression
    compressed_text, word_to_id = compress_text(text)
    compressed_size = sys.getsizeof(pickle.dumps(compressed_text)) + sys.getsizeof(pickle.dumps(word_to_id))
    print(f"Compressed size: {compressed_size} bytes")

    # Save compressed data to files
    with open("compressed.pkl", "wb") as f:
        pickle.dump(compressed_text, f)
    with open("dictionary.pkl", "wb") as f:
        pickle.dump(word_to_id, f)

    # Load compressed data from files
    with open("compressed.pkl", "rb") as f:
        loaded_compressed_text = pickle.load(f)
    with open("dictionary.pkl", "rb") as f:
        loaded_word_to_id = pickle.load(f)

    # Decompression
    decompressed_text = decompress_text(loaded_compressed_text, loaded_word_to_id)
    decompressed_size = sys.getsizeof(decompressed_text)
    print(f"Decompressed size: {decompressed_size} bytes (should match original size)")

    # Efficiency
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio}")
