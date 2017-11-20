# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:58:52 2017

@author: rd0348
"""
import tensorflow as tf
import numpy as np

def test_word2vec():

    dataset = type('dummy', (), {})()     #create a dynamic object and then add attributes to it
    def dummySampleTokenIdx():            #generate 1 integer between (0,4)
        return np.random.randint(0, 4)

    def getRandomContext(C):                            #getRandomContext(3) = ('d', ['d', 'd', 'd', 'e', 'a', 'd'])
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[np.random.randint(0,4)], [tokens[np.random.randint(0,4)] \
            for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx        #add two methods to dataset
    dataset.getRandomContext = getRandomContext

    np.random.seed(31415)                                    
    np.random.seed(9265)                                #can be called again to re-seed the generator

    #in this test, this wordvectors matrix is randomly generated,
    #but in real training, this matrix is a well trained data
    dummy_vectors = normalizeRows(np.random.randn(10,3))                    #generate matrix in shape=(10,3), 
    dummy_tokens = dict([("a",0), ("b",1), ("c",2), ("d",3), ("e",4)])        #{'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)  #vec is dummy_vectors

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))


if __name__ == "__main__":
    test_word2vec()
