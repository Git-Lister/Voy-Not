import linecache

for i in range(285, 295):
    print(f'{i}: {linecache.getline("src/layer3_mechanisms/evaluate.py", i).rstrip()}')