import dataset
import generate_test_splits
import nltk
sentence = "At eight o'clock on Thursday morning... Arthur didn't feel very good."
tokens = nltk.word_tokenize(sentence)


dataset = DataSet()
generate_test_splits.generate_hold_out_split(dataset, training = 0.9, base_dir="splits")