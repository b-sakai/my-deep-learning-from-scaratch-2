import sys
sys.path.append('../')
from common.util import most_similar, analogy
import pickle

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

querys = ['you', 'year', 'car', 'toyota']   
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)
analogy('east', 'eastern', 'west', word_to_id, id_to_word, word_vecs)
analogy('up', 'upward', 'down', word_to_id, id_to_word, word_vecs)
analogy('land', 'landed', 'water', word_to_id, id_to_word, word_vecs)
analogy('work', 'working', 'walk', word_to_id, id_to_word, word_vecs)
analogy('paint', 'painting', 'read', word_to_id, id_to_word, word_vecs)
analogy('fly', 'flew', 'eat', word_to_id, id_to_word, word_vecs)
