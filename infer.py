import heapq
#from itertools import izip
import numpy as np

def ugly_normalize(vecs):
   normalizers = np.sqrt((vecs * vecs).sum(axis=1))
   normalizers[normalizers==0]=1
   return (vecs.T / normalizers).T

class Embeddings:
   def __init__(self, vecsfile, vocabfile=None, normalize=True, fMode=True):
      if fMode:
        if vocabfile is None: vocabfile = vecsfile.replace("npy","vocab")
        self._vecs = np.load(vecsfile)
        self._vocab = open(vocabfile).read().split()
      else:
        self._vecs = vecsfile[0]
        self._vocab = vecsfile[1]
      if normalize:
         self._vecs = ugly_normalize(self._vecs)
      self._w2v = {w:i for i,w in enumerate(self._vocab)}

   @classmethod
   def load(cls, vecsfile, vocabfile=None):
      return Embeddings(vecsfile, vocabfile)

   def word2vec(self, w):
      return self._vecs[self._w2v[w]]

   def similar_to_vec(self, v, N=10):
      sims = self._vecs.dot(v)
      sims = heapq.nlargest(N, list(zip(sims,self._vocab,self._vecs)))
      return sims

   def similar_to_avg(self, v,w):
      sims = self._vecs[self._w2v[w]].dot(v)
      return sims


   def similarity(self,word1,word2):
      vecs = self._vecs
      if word1 not in self._w2v:
        return 0
      if word2 not in self._w2v:
        return 0
      w1 = self._w2v[word1]
      w2 = self._w2v[word2]
      return vecs[w1].dot(vecs[w2])

   def most_similar(self, word, N=10):
      w = self._vocab.index(word)
      sims = self._vecs.dot(self._vecs[w])
      sims = heapq.nlargest(N, list(zip(sims,self._vocab)))
      return sims

   def scoreAnalogy(self,a1,b1,c1,d1,mult=True):
     # for target d, how much like a and b is c and d?
     wvecs = self._vecs
     if a1 not in self._w2v:
       return 0
     if b1 not in self._w2v:
       return 0
     if c1 not in self._w2v:
       return 0
     if d1 not in self._w2v:
       return 0
     d = wvecs[self._w2v[d1]]

     if mult:
       a = (1+d.dot(wvecs[self._w2v[a1]]))/2
       b = (1+d.dot(wvecs[self._w2v[b1]]))/2
       c = (1+d.dot(wvecs[self._w2v[c1]]))/2
       return c*b/a

     else:
       a = d.dot(wvecs[self._w2v[a1]])
       b = d.dot(wvecs[self._w2v[b1]])
       c = d.dot(wvecs[self._w2v[c1]])
       return c+b-a

   def analogy(self, pos1, neg1, pos2,N=10,mult=True):
      wvecs, vocab = self._vecs, self._vocab
      p1 = vocab.index(pos1)
      p2 = vocab.index(pos2)
      n1 = vocab.index(neg1)
      if mult:
         p1,p2,n1 = [(1+wvecs.dot(wvecs[i]))/2 for i in (p1,p2,n1)]
         if N == 1:
            return max(((v,w) for v,w in zip((p1 * p2 / n1),vocab) if w not in [pos1,pos2,neg1]))
         return heapq.nlargest(N,((v,w) for v,w in zip((p1 * p2 / n1),vocab) if w not in [pos1,pos2,neg1]))
      else:
         p1,p2,n1 = [(wvecs.dot(wvecs[i])) for i in (p1,p2,n1)]
         if N == 1:
            return max(((v,w) for v,w in zip((p1 + p2 - n1),vocab) if w not in [pos1,pos2,neg1]))
         return heapq.nlargest(N,((v,w) for v,w in zip((p1 + p2 - n1),vocab) if w not in [pos1,pos2,neg1]))

if __name__ == '__main__':
   import sys

   e = Embeddings.load(sys.argv[1])

   #print e.most_similar('azkaban')
   #print e.analogy('king','man','woman')
   print (e.most_similar('azkaban'))
   print (e.analogy('king','man','woman'))
