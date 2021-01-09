import string

ret = """graph LR

"""

nodes = [char for char in string.ascii_uppercase + string.ascii_lowercase + "0123456789"]
nodes.append('start')
nodes.append('finish')
print(nodes)
for symbol in nodes:
    if(symbol in ["\t", "(", ")", "\n", "\r", "\x0b", "\x0c", " "]):
        continue
    ret+=f"    {symbol}(({symbol}))\n"

for src in nodes:
    for target in nodes:
        if(target=='start'): continue
        if(src=='end'): continue
        ret+=f"    {src} --> {target}\n"

open('documentation/includes/full_graph.html', "w").write(ret)