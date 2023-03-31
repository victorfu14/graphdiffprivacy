def process(file):
    graph = {}
    with open(file) as f:
        lines = f.readlines()
    f.close()
    return lines
