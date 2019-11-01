import sys
from Levenshtein import distance

ref = []
with open(sys.argv[1]) as file1:
    for line in file1:
        ref.append(line.strip())

pred = []
with open(sys.argv[2]) as file2:
    for line in file2:
        pred.append(line.strip())

assert len(ref) == len(pred)
dist = []
for x in range(len(ref)):
    dist.append(distance(ref[x], pred[x]))

print("avg edit distance:", sum(dist) / len(dist))
