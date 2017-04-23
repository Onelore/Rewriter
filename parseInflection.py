import sys
import os
from collections import defaultdict

inflections = [x.strip().split() for x in open("inflection.txt").readlines() if "{" not in x and "~" not in x]
infld = defaultdict(list)
for i,x in enumerate(inflections):
  for w in x:
    infld[w].append(i)

def get(w,pos):
  pos = pos.lower()
  #opts = [x for x in inflections if (" "+w+" " in x or x[:len(w)+1]==w+" " or x.strip()[-(len(w)+1):]==" "+w)
  #    and ("{" not in x and "~" not in x)]
  #if not opts:
  #  return None
  #opts2 = [x for x in opts if x.split(":")[0][-1].lower()==pos[0]]
  #if opts2:
  #  opts = opts2
  opts = [inflections[i] for i in infld[w]]
  opts = [x for x in opts if x[1][0].lower()==pos[0]]
  if not opts: return w

  pick = opts[0]
  pick = [x for x in pick if "(" not in x and "@" not in x and ")" not in x]
  if len(pick)<3:
    return pick[0]
  if pos=='nns' or pos=='vbd':
    return pick[2]
  if pos=='nn' or pos=='vb' or pos=='vbp':
    return pick[0]
  if pos=='vbz':
    return pick[-1]
  if pos=='vbg':
    return pick[-2]
  if pos=='vbn':
    if len(pick)>5:
      return pick[3]
    else:
      return pick[2]

  return w

  

if __name__=="__main__":
  print(get("cow",'nn'))
  print(get("cow",'nns'))
  print(get("cw",'nn'))
  print(get("walks",'vb'))
  print(get("walks",'vbn'))
  print(get("walking",'vbg'))
  print(get("eat",'vbn'))
  print(get("eat",'vbg'))
  print(get("ate",'vb'))
