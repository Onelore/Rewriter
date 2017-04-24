import sys, os, csv
d = sys.argv[1]
if d[-1] == '/':
  d = d[:-1]
c = csv.reader(open(d+"/"+d+".csv"))
g = open(d+"/gold.txt",'w')
f = open(d+"/orig.txt",'w')
i = 0
for l in list(c)[1:]:
  g.write(l[-1].replace("\n",' ')+'\n')
  if i%3==0:
    f.write(l[-4]+'\n')
  i+=1

