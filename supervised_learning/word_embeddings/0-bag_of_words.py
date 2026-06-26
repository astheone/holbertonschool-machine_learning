#!/usr/bin/env python3
"""
Modul për krijimin e një matrice Bag of Words (BoW)
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Krijon një matricë embedding duke përdorur Bag of Words.

    Args:
        sentences: Një listë me fjali për t'u analizuar.
        vocab: Një listë me fjalët e fjalorit (opsionale).

    Returns:
        embeddings: Një numpy.ndarray me formë (s, f) që përmban embeddings.
        features: Një listë e fjalëve (features) të përdorura.
    """
    # 1. Pastrimi i fjalive dhe tokenizimi (kthehen në fjalë të vogla pa pikësim)
    cleaned_sentences = []
    all_words = set()

    for sentence in sentences:
        # Heqim shenjat e pikësimit dhe mbajmë vetëm fjalët/shkronjat
        tokens = re.sub(r'[^\w\s]', '', sentence).lower().split()
        cleaned_sentences.append(tokens)
        if vocab is None:
            all_words.update(tokens)

    # 2. Përcaktimi i fjalorit (features)
    if vocab is not None:
        features = vocab
    else:
        features = sorted(list(all_words))

    # Krijojmë një indeks për çdo fjalë në fjalor për akses më të shpejtë
    vocab_index = {word: i for i, word in enumerate(features)}

    # 3. Ndërtimi i matricës së embeddings
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    for i, tokens in enumerate(cleaned_sentences):
        for token in tokens:
            if token in vocab_index:
                embeddings[i, vocab_index[token]] += 1

    return embeddings, features
