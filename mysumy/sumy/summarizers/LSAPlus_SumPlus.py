import numpy as np
from nltk import sent_tokenize, word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from mysumy.sumy.summarizers._summarizer import AbstractSummarizer
from mysumy.sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
from numpy.linalg import svd as singular_value_decomposition


#### Trims the Original Text of Unimportant Sentences which are highly indicated
#### Returns a list of trimmed texts

def LSAPlus_SumPlus(doc):

    # SumPlus
    sumbasic_sents = []
    for text in doc:
        tsummarizer_w_stops = SumBasicSummarizer()
        tsummarizer_w_stops.stop_words = get_stop_words('english')
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        dictionary =  tsummarizer_w_stops._compute_ratings(parser.document.sentences)
        sumbasic_sents_entries = []
        for sent in dictionary:
            sumbasic_sents_entries.append(sent)
        sumbasic_sents.append(sumbasic_sents_entries)

    #LSAPlus
    
    lsa_sents = []
    for text in doc:
        l2summarizer = LsaSummarizer()
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        dictionary = (l2summarizer._create_dictionary(parser.document))
        matrix = l2summarizer._create_matrix(parser.document, dictionary)
        matrix2 = l2summarizer._compute_term_frequency(matrix)
        u, sigma, v = singular_value_decomposition(matrix2, full_matrices=False)
        v_sorted = sorted(abs(v[:, 0]), reverse = True)
        v_indices = []
        for i in v_sorted:
            v_indices.append(list(v_sorted).index(i))

        sents = np.array(list(parser.document.sentences))
        sents[np.array(v_indices)]
        lsa_sents_entries = list(sents)
        lsa_sents.append(lsa_sents_entries)

    # Combining SumPlus and LSAPlus
    import math
    num_sentences = len(sumbasic_sents)
    all_sents_removed_parent2 = []
    for entry in range(num_sentences):
        num_sents_to_remove = math.ceil(len(sumbasic_sents[entry])/2) 
        sent_len = len(sumbasic_sents[entry])
        sb = sumbasic_sents[entry][sent_len - num_sents_to_remove: sent_len]
        lsa = lsa_sents[entry][sent_len - num_sents_to_remove: sent_len]

        # Checking if Sentences are ranked bad by BOTH LSAPlus and SumPlus
        sents_removed3 = []
        for sent in lsa:
            if (sent in sb):
                sents_removed3.append(sent)  

        # Setences to be Trimmed Off
        all_sents_removed_parent2.append(sents_removed3)

    sents_to_keep_parent2 = []
    for i in range(len(doc)):
        parser = PlaintextParser.from_string(doc[i], Tokenizer('english'))
        sents = parser.document.sentences

        # Sentences not Trimmed Off
        sents_to_keep2 = [sentence for sentence in sents if sentence not in all_sents_removed_parent2[i]]
        
        # Appending Trimmed Text for Each Entry
        sents_to_keep_parent2.append(sents_to_keep2)

    
    # Trimmed Text
    sentence_parent2 = []
    for text in sents_to_keep_parent2:
        sentence = ""
        for sent in text:
            sentence = sentence + " " + str(sent)
        sentence_parent2.append(sentence)
                     
    return sentence_parent2
    