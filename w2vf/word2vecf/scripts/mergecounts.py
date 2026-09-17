import sys
from collections import defaultdict
wc = defaultdict(int)
for line in sys.stdin:
   line = line.strip().split()
   if len(line)!=2:
      print("skipping line",line, file=sys.stderr)
      continue
   wc[line[0]]+=int(line[1])
for w,c in sorted(iter(wc.items()),key=lambda x:-x[1]):
   print(w,c)


