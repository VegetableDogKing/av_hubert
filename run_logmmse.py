import os
from oct2py import Oct2Py
from tqdm.contrib import tzip

input_lst = "noisy.lst"
out_lst = "out.lst"

oc = Oct2Py()

with open(input_lst) as file:
    lines = file.readlines()
    input_lines = [line.rstrip() for line in lines]

with open(out_lst) as file:
    lines = file.readlines()
    output_lines = [line.rstrip() for line in lines]


for (noisy, enhan) in tzip(input_lines, output_lines):
    directory = os.path.dirname(enhan)
    if not os.path.exists(directory):
        os.makedirs(directory)
    oc.logmmse(noisy,enhan)

