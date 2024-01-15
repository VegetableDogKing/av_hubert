

path = "./baseline_result/logmmse/"
out = "./baseline_result/logmmse.txt"


import os
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(path):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]



with open(out, 'w') as f:
    for item in listOfFiles:
        f.write("%s\n" % item)