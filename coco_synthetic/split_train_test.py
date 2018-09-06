import networkx as nx
import numpy as np
import os
from glob import glob 
def contain_node(Graph_list,node):
	for g in Graph_list:
		if g.has_node(node):
			return True 
	return False
data_dir='./cocostuff/coco/filter_tamper'
ext='Tp*'

#cls=['person','tv','airplane','dog','bench','train','kite','bed','refrigerator','bowl']
cls=['person','airplane','dog','train','bed','refrigerator']
filenames=glob(os.path.join(data_dir,ext))

G=nx.Graph()
print(len(filenames))
for file in filenames:
	content=os.path.splitext(os.path.basename(file))[0].split("_")
	if content[-1] in cls:
		target_name=content[1]
		source_name=content[2]
		G.add_edge(target_name,source_name)

train = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0:950]
test=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[950:]


with open('train_filter.txt','w') as f:
	for file in filenames:
		content=os.path.splitext(os.path.basename(file))[0].split("_")
		if content[-1] in cls:
			target_name=content[1]
			source_name=content[2]
			if target_name!= source_name and contain_node(train,target_name) and contain_node(train,source_name):
				f.write('%s\n' % file) 

 

with open('test_filter.txt','w') as f:
	for file in filenames:
		content=os.path.splitext(os.path.basename(file))[0].split("_")
		if content[-1] in cls:
			target_name=content[1]
			source_name=content[2]
			if target_name!= source_name and contain_node(test,target_name) and contain_node(test,source_name):
				f.write('%s\n' % file) 





