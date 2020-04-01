import numpy as np

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    g_vectors = {}
    words = set()
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        g_vectors[word] = embedding
        words.add(word)

    
    words_to_index = {}
    index_to_words = {}
    i = 1
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i += 1
    print("Done.",len(g_vectors)," words loaded!")
    return g_vectors, words_to_index, index_to_words
