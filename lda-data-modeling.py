from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora

# Latent Dirichlet Allocation for Topic Modeling
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."

# compile documents
doc_complete = [doc1, doc2, doc3, doc4, doc5]

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]

"""
[[u'sugar', u'bad', u'consume', u'sister', u'like', u'sugar', u'father'],
 ['father', 'spends', 'lot', 'time', 'driving', 'sister', 'around', 'dance', 'practice'],
  [u'doctor', u'suggest', u'driving', u'may', u'cause', u'increased', u'stress', u'blood', u'pressure'],
   ['sometimes', 'feel', 'pressure', 'perform', 'well', 'school',
    'father', 'never', 'seems', 'drive', 'sister', 'better'],
     [u'health', u'expert', u'say', u'sugar', u'good', u'lifestyle']]
"""


# Creating the term dictionary of our courpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
"""
[[(0, 1), (1, 1), (2, 1), (3, 2), (4, 1), (5, 1)], [(0, 1), (2, 1), (6, 1), (7, 1), (8, 1), (9, 1),
 (10, 1), (11, 1), (12, 1)], [(7, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1),
  (20, 1)], [(0, 1), (2, 1), (18, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1),
   (28, 1), (29, 1)], [(3, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1)]]
"""

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=3, num_words=3))
"""
[(0, u'0.059*"pressure" + 0.059*"father" + 0.059*"sister"'),
 (1, u'0.065*"driving" + 0.065*"father" + 0.065*"sister"'),
 (2, u'0.076*"sugar" + 0.075*"expert" + 0.075*"good"')]
"""