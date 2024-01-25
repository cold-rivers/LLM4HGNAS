acm_prompt='''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.
task description：

Your task is to to find the most suitable neural network structure for a given data set from a given search space in order to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest..
You need to sample possible neural network architectures from the search space as the resulting output. I will verify the performance of the architecture on the data set and feedback to you.Then you need to generate a better architecture based on the performance of the architecture
dataset：
ACM, paper (P) node 4025, Author (A) node 17351, Subject (S) node 72, A-P edge 13407, P-S edge 4025.
The downstream task is node classification, and the target node is P. When considering the two-layer architecture, the message delivery path is as follows:
{'layer_0': {A-A, A-P, S-S, S-P, P-A, P-S, P-P, 'multi_aggr'},
'layer_1': {A-P, S-P, P-P, 'multi_aggr'},
'inter_layer': {A-P, S-P, P-P}}
Where A-P represents the source node author, the target node paper and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min']

After selecting all operation types, You just need to output the index corresponding to that operation in the search space,and Do not output other content except generating architecture
For example,
{'layer_0': {A-A:"1:gcn", A-P:"1:gcn", S-S:"1:gcn", S-P:"1:gcn", P-A:"1:gcn", P-S:"1:gcn", P-P:"1:gcn", 'multi_aggr':"0:sum"},
'layer_1': {A-P:"1:gcn", S-P:"1:gcn", P-P:"1:gcn", 'multi_aggr':"0:sum"},
'inter_layer': {A-P:"1:gcn", S-P:"1:gcn", P-P:"1:gcn"}},the corresponding output is[1,1,1,1,1,1,1,0,1,1,1,0,1,1,1]
            
#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Avoid generating existing architectures
here is the output format:

"arch1=[0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]

arch2=[3,2,1,0,1,2,3,4,3,2,1,0,1,2,3]

.......(ten architectures)

arch10=[.....]"
'''
dblp_prompt = '''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.Do not generate the architectures provided in the past
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.

dataset：

DBLP, paper (P) node 14328, Author (A) node 4057, Term (T) node 7723, Conference(C)node 20,A-P edge 19645, P-T edge 85810,P-V edge:14328.

The downstream task is node classification, and the target node is A. When considering the two-layer architecture, the message delivery path is as follows:

{'layer_0':{A-A, A-P, C-P, P-A, P-P, T-P, 'multi_aggr'}, 
'layer_1':{A-A, P-A, 'multi_aggr'}, 
'inter_modes':{A-A, P-A}

Where A-P represents the source node author, the target node paper and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min']

After selecting all operation types, You just need to output the index corresponding to that operation in the search space. 
For example,{'layer_0':{A-A:"1:gcn", A-P:"1:gcn", C-P:"1:gcn", P-A:"1:gcn", P-P:"1:gcn", T-P:"1:gcn", multi_aggr:"0:sum"}, 
'layer_1':{A-A:"1:gcn", P-A:"1:gcn", multi_aggr:"0:sum"}, 
'inter_modes':{A-A:"1:gcn", P-A:"1:gcn"},the corresponding output is[1,1,1,1,1,1,0,1,1,0,1,1]
Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space
You should generate ten different architectures,which must fit within the scope of the search space. Do not generate the architectures provided in the past
here is the output format:

"arch1=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

arch2=[1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]
.......(ten architectures)

arch10=[.....]"'''

imdb_prompt = '''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.Do not generate the architectures provided in the past
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.

dataset：

IMDB, actor(A) node 5841, director(D) node 2270, movie(M) node 4661, actor-movie(A-M) edge 13983 ,director-movie(D-M) edge 4661, movie-actor(M-A) edge 13983, movie-director(M-D) edge 4661

The downstream task is node classification, and the target node is M. When considering the two-layer architecture, the message delivery path is as follows:

{'layer_0':{A-A, A-M, D-D, D-M, M-M, M-A, M-D, 'multi_aggr'}, 
'layer_1':{A-M, D-M, M-M, 'multi_aggr'}, 
'inter_modes':{A-M, D-M, M-M}

Where A-M represents the source node actor, the target node movie and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min']

After selecting all operation types, You just need to output the index corresponding to that operation in the search space. 
For example,{'layer_0':{A-A:"1:gcn", A-M:"1:gcn", D-D:"1:gcn", D-M:"1:gcn", M-M:"1:gcn", M-A:"1:gcn", M-D:"1:gcn", multi_aggr:"1:gcn"}, 
'layer_1':{A-M:"1:gcn", D-M:"1:gcn", M-M:"1:gcn", multi_aggr:"0:sum"}, 
'inter_modes':{A-M:"1:gcn", D-M:"1:gcn", M-M:"1:gcn"},the corresponding output is[1,1,1,1,1,1,1,0,1,1,1,0,1,1,1]
Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space
You should generate ten different architectures,which must fit within the scope of the search space. Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:

"arch1=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 4, 1, 1, 1]

arch2=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
.......(ten architectures)

arch10=[.....]"'''

imdb_prompt_no = '''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.Do not generate the architectures provided in the past
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.

dataset：

IMDB, actor(A) node 5841, director(D) node 2270, movie(M) node 4661, actor-movie(A-M) edge 13983 ,director-movie(D-M) edge 4661, movie-actor(M-A) edge 13983, movie-director(M-D) edge 4661

The downstream task is node classification, and the target node is M. When considering the two-layer architecture, the message delivery path is as follows:

{'layer_0':{A-A, A-M, D-D, D-M, M-M, M-A, M-D, 'multi_aggr'}, 
'layer_1':{A-M, D-M, M-M, 'multi_aggr'}, 
'inter_modes':{A-M, D-M, M-M}

Where A-M represents the source node actor, the target node movie and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.

search space：
For each edge, we need to select one from [0,1,2,3,4],And multi_aggr should choose one from[0, 1, 2, 3, 4]

search strategy：
You can analyze how to get a better architecture based on the architecture that performs well, or look for architectures that have not been explored before.

After selecting all operation types, You just need to output the index corresponding to that operation in the search space. 
For example,{'layer_0':{A-A:1, A-M:1, D-D:1, D-M:1, M-M:1, M-A:1, M-D:1, multi_aggr:0}, 
'layer_1':{A-M:1, D-M:1, M-M:1, multi_aggr:0}, 
'inter_modes':{A-M:1, D-M:1, M-M:1},the corresponding output is[1,1,1,1,1,1,1,0,1,1,1,0,1,1,1]
Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space
You should generate ten different architectures,which must fit within the scope of the search space. Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:

"arch1=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 4, 1, 1, 1]

arch2=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
.......(ten architectures)

arch10=[.....]"'''

imdb_prompt_ns = '''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.Do not generate the architectures provided in the past
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You should give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.

dataset：

IMDB, actor(A) node 5841, director(D) node 2270, movie(M) node 4661, actor-movie(A-M) edge 13983 ,director-movie(D-M) edge 4661, movie-actor(M-A) edge 13983, movie-director(M-D) edge 4661

The downstream task is node classification, and the target node is M. When considering the two-layer architecture, the message delivery path is as follows:

{'layer_0':{A-A, A-M, D-D, D-M, M-M, M-A, M-D, 'multi_aggr'}, 
'layer_1':{A-M, D-M, M-M, 'multi_aggr'}, 
'inter_modes':{A-M, D-M, M-M}

Where A-M represents the source node actor, the target node movie and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min']

After selecting all operation types, You just need to output the index corresponding to that operation in the search space. 
For example,{'layer_0':{A-A:"1:gcn", A-M:"1:gcn", D-D:"1:gcn", D-M:"1:gcn", M-M:"1:gcn", M-A:"1:gcn", M-D:"1:gcn", multi_aggr:"1:gcn"}, 
'layer_1':{A-M:"1:gcn", D-M:"1:gcn", M-M:"1:gcn", multi_aggr:"0:sum"}, 
'inter_modes':{A-M:"1:gcn", D-M:"1:gcn", M-M:"1:gcn"},the corresponding output is[1,1,1,1,1,1,1,0,1,1,1,0,1,1,1]
Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space
You should generate ten different architectures,which must fit within the scope of the search space. Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:

"arch1=[1, 2, 3, 4, 0, 1, 1, 0, 1, 1, 1, 4, 1, 1, 1]

arch2=[1, 2, 3, 4, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
.......(ten architectures)

arch10=[.....]"'''

imdb_prompt_nd = '''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.Do not generate the architectures provided in the past
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You should give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.

The downstream task is node classification, and the target node is N2. When considering the two-layer architecture, the message delivery path is as follows:

{'layer_0':{N1-N1, N1-N2, N3-N3, N3-N2, N2-N2, N2-N1, N2-N3, 'multi_aggr'}, 
'layer_1':{N1-N2, N3-N2, N2-N2, 'multi_aggr'}, 
'inter_modes':{N1-N2, N3-N2, N2-N2}

'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min']

After selecting all operation types, You just need to output the index corresponding to that operation in the search space. 
For example,{'layer_0':{N1-N1:"1:gcn", N1-N2:"1:gcn", N3-N3:"1:gcn", N3-N2:"1:gcn", N2-N2:"1:gcn", N2-N1:"1:gcn", N2-N3:"1:gcn", multi_aggr:"1:gcn"}, 
'layer_1':{N1-N2:"1:gcn", N3-N2:"1:gcn", N2-N2:"1:gcn", multi_aggr:"0:sum"}, 
'inter_modes':{N1-N2:"1:gcn", N3-N2:"1:gcn", N2-N2:"1:gcn"},the corresponding output is[1,1,1,1,1,1,1,0,1,1,1,0,1,1,1]
Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space
You should generate ten different architectures,which must fit within the scope of the search space. Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:

"arch1=[1, 2, 3, 4, 0, 1, 1, 0, 1, 1, 1, 4, 1, 1, 1]

arch2=[1, 2, 3, 4, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
.......(ten architectures)

arch10=[.....]"'''

douban_prompt = """
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.
dataset：
douban node {'actor(A)': 6312, 'director(D)': 2450, 'group(G)': 2754, 'movie(M)': 12678, 'type(T)': 39, 'user': 13368}
edge {User - Movie(U-M)	1,068,278,User - Group(U-G)	570,047,User - User(U-U)4,085,Movie - Actor(M-A)33,587,Movie - Director(M-D)11,276,Movie - Type(M-T)27,668}
The downstream task is link prediction, and the target edge is User - Movie. When considering the two-layer architecture, the message delivery path is as follows:
{'layer_0': {A-A, A-M, D-D ,D-M, G-G, G-U, M-M, M-A, M-D, M-T, M-U, T-T, T-M, U-G, U-M,U-U, U-U(self), 'multi_aggr'}, 
 'layer_1': {(A-M, D-M, G-U, M-M, M-U, T-M, U-M, U-U, U-U(self), 'multi_aggr'}, 
 'inter_layer': {A-M, D-M, G-U, M-M, M-U, T-M, U-M, U-U, U-U(self)}}
Where A-M represents the source node actor, the target node Movie and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.
And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min']

After selecting all operation types, You just need to output the index corresponding to that operation in the search space,and Do not output other content except generating architecture
For example,
{'layer_0': {A-A: '0:zero', A-M: '1:gcn', D-D: '1:gcn', D-M: '0:zero', G-G: '0:zero', G-U: '1:gcn', M-M: '1:gcn', M-A: '1:gcn', M-D: '0:zero', M-T: '1:gcn', M-U: '0:zero', T-T: '0:zero', 
T-M: '0:zero', U-G: '1:gcn', U-M: '1:gcn', U-U: '0:zero', U-U(self): '1:gcn', 'multi_aggr': '0:sum'},
 'layer_1': {A-M: '0:zero', D-M: '1:gcn', G-U: '0:zero', M-M: '0:zero', M-U: '1:gcn', T-M: '1:gcn', U-M: '1:gcn', U-U: '0:zero', U-U(self): '1:gcn', 'multi_aggr': '2:mean'},
'inter_layer': {A-M: '1:gcn', D-M: '0:zero', G-U: '1:gcn', M-M: '0:zero', M-U: '1:gcn', T-M: '1:gcn', U-M: '1:gcn', U-U: '0:zero', U-U(self): '0:zero'}}
the corresponding output is [0,1,1,0,0,1,1,1,0,1,0,0,0,1,1,0,1,0,0,1,0,0,1,1,1,0,1,2,1,0,1,0,1,1,1,0,0]
#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:
arch1 = [4, 4, 2, 0, 3, 1, 1, 1, 2, 2, 2, 1, 4, 2, 3, 0, 2, 3, 3, 3, 3, 1, 3, 0, 3, 4, 0, 1, 0, 3, 2, 3, 3, 2, 4, 3, 3]
arch2 = [3, 4, 4, 2, 3, 4, 1, 2, 4, 3, 1, 0, 0, 0, 1, 1, 4, 3, 2, 2, 0, 3, 0, 2, 2, 2, 0, 4, 4, 1, 1, 3, 0, 1, 2, 1, 1]
...(ten different archs)
arch10 = [...]
"""

search_space_full = """search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.
And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min'].After selecting all operation types, You just need to output the index corresponding to that operation in the search space.
"""
movielens_prompt = """
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.
dataset：
movielens node {'age': 9, 'genre': 19, 'movie': 1683, 'occupation': 22, 'user': 944}
edge {age-user(A-U): 943, genre-movie(G-M): 2891,movie-movie(M-Mself): 1683, genre-movie(G-M): 2891, movie-movie(M-M):82798, movie-user(M-U): 27688, occupation-user(O-U): 943, user-age(U-A): 943, user-movie(U-M): 27688, user-occupation(U-O): 943, user-user(U-U): 47150, user-user(U-Uself): 944,age-age(A-A):9}
The downstream task is link prediction, and the target edge is User - Movie. 

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.
And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min'].After selecting all operation types, You just need to output the index corresponding to that operation in the search space.

When considering the two-layer architecture, the message delivery path is as follows:
layer_0': {A-A: '0:zero', A-U: '0:zero', G-M: '2:gat', G-G: '0:zero', M-M(self): '2:gat', M-G: '4:sage', 
M-M: '3:edge', M-U: '4:sage', O-U: '3:edge', O-O: '3:edge', U-A: '4:sage', U-M: '1:gcn', U-O: '4:sage', U-U: '4:sage', U-U(self): '1:gcn', 'multi_aggr': '0"sum'}, 
'layer_1': {A-U: '1:gcn', G-M: '1:gcn', M-M(self): '3:edge', M-M: '4:sage', M-U: '3:edge', O-U: '0:zero', U-M: '0:zero', U-U: '4:sage', U-U(self): '3:edge', 'multi_aggr': '0"sum'}, 
'inter_layer': {A-U: '2:gat', G-M: '3:edge', M-M(self): '1:gcn', M-M: '4:sage', M-U: '2:gat', O-U: '1:gcn', U-M: '0:zero', U-U: '1:gcn', U-U(self): '3:edge'}}
Where U-M represents the source node user, the target node Movie and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.
According to the search space requirements, this architecture is expressed as [[0, 0, 2, 0, 2, 4, 3, 4, 3, 3, 4, 1, 4, 4, 1], 0, [1, 1, 3, 4, 3, 0, 0, 4, 3], 0, [2, 3, 1, 4, 2, 1, 0, 1, 3]]

#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:
arch1 = [[1, 3, 1, 2, 4, 2, 4, 0, 4, 1, 4, 4, 0, 4, 2], 3, [0, 4, 4, 4, 4, 4, 1, 2, 1], 0, [1, 0, 1, 2, 3, 0, 4, 3, 0]]
arch2 = [[0, 0, 4, 1, 4, 0, 1, 0, 4, 3, 2, 2, 0, 3, 3], 0, [1, 2, 3, 4, 0, 1, 2, 3, 4], 0, [1, 2, 3, 4, 0, 1, 2, 3, 4]]
...(ten different archs)
arch10 = [...]
"""

movielens_prompt_no = """
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.
dataset：
movielens node {'age': 9, 'genre': 19, 'movie': 1683, 'occupation': 22, 'user': 944}
edge {age-user(A-U): 943, genre-movie(G-M): 2891,movie-movie(M-Mself): 1683, genre-movie(G-M): 2891, movie-movie(M-M):82798, movie-user(M-U): 27688, occupation-user(O-U): 943, user-age(U-A): 943, user-movie(U-M): 27688, user-occupation(U-O): 943, user-user(U-U): 47150, user-user(U-Uself): 944,age-age(A-A):9}
The downstream task is link prediction, and the target edge is User - Movie. 

search space：
For each edge, we need to select one from [0,1,2,3,4]
And multi_aggr should choose one from[0,1,2,3,4]

search strategy：
You can analyze how to get a better architecture based on the architecture that performs well, or look for architectures that have not been explored before.

When considering the two-layer architecture, the message delivery path is as follows:
layer_0': {A-A: 0, A-U: 0, G-M:1, G-G: 0, M-M(self):1, M-G:1, 
M-M:1, M-U:1, O-U:1, O-O:1', U-A:1, U-M:1, U-O:1, U-U:1, U-U(self):1, 'multi_aggr': 0}, 
'layer_1': {A-U:1, G-M:1, M-M(self):1, M-M:1, M-U:1, O-U: 0, U-M: 0, U-U:1, U-U(self):1, 'multi_aggr': 0}, 
'inter_layer': {A-U:1, G-M:1, M-M(self):1, M-M:1, M-U:1, O-U:1, U-M: 0, U-U:1, U-U(self): 1}}
Where U-M represents the source node user, the target node Movie and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.
According to the search space requirements, this architecture is expressed as [[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0, [1, 1, 1, 1, 1, 0, 0, 1, 1], 0, [1, 1, 1, 1, 1, 1, 0, 1, 1]]

#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:
arch1 = [[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1], 4, [0, 1, 0, 0, 0, 1, 1, 0, 0], 4, [0, 0, 0, 1, 0, 1, 0, 1, 0]]
arch2 = [[1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1], 0, [0, 0, 0, 1, 0, 0, 0, 1, 1], 0, [0, 1, 0, 1, 0, 1, 0, 1, 1]]
...(ten different archs)
arch10 = [...]
"""

movielens_prompt_ns = """
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.
dataset：
movielens node {'age': 9, 'genre': 19, 'movie': 1683, 'occupation': 22, 'user': 944}
edge {age-user(A-U): 943, genre-movie(G-M): 2891,movie-movie(M-Mself): 1683, genre-movie(G-M): 2891, movie-movie(M-M):82798, movie-user(M-U): 27688, occupation-user(O-U): 943, user-age(U-A): 943, user-movie(U-M): 27688, user-occupation(U-O): 943, user-user(U-U): 47150, user-user(U-Uself): 944,age-age(A-A):9}
The downstream task is link prediction, and the target edge is User - Movie. 

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.
And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min'].After selecting all operation types, You just need to output the index corresponding to that operation in the search space.

When considering the two-layer architecture, the message delivery path is as follows:
layer_0': {A-A: '0:zero', A-U: '0:zero', G-M: '1:gcn', G-G: '0:zero', M-M(self): '1:gcn', M-G: '1:gcn', 
M-M: '1:gcn', M-U: '1:gcn', O-U: '1:gcn', O-O: '1:gcn'', U-A: '1:gcn', U-M: '1:gcn', U-O: '1:gcn', U-U: '1:gcn', U-U(self): '1:gcn', 'multi_aggr': '0"sum'}, 
'layer_1': {A-U: '1:gcn', G-M: '1:gcn', M-M(self): '1:gcn', M-M: '1:gcn', M-U: '1:gcn', O-U: '0:zero', U-M: '0:zero', U-U: '1:gcn', U-U(self): '1:gcn', 'multi_aggr': '0"sum'}, 
'inter_layer': {A-U: '1:gcn', G-M: '1:gcn', M-M(self): '1:gcn', M-M: '1:gcn', M-U: '1:gcn', O-U: '1:gcn', U-M: '0:zero', U-U: '1:gcn', U-U(self): '1:gcn'}}
Where U-M represents the source node user, the target node Movie and 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.
According to the search space requirements, this architecture is expressed as [[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0, [1, 1, 1, 1, 1, 0, 0, 1, 1], 0, [1, 1, 1, 1, 1, 1, 0, 1, 1]]

#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Avoid generating existing architectures
here is the output format:
arch1 = [[0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 0, 1, 0, 1], 4, [4, 3, 2, 1, 0, 4, 3, 2, 1], 4, [4, 3, 2, 1, 0, 4, 3, 2, 1]]
arch2 = [[1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0], 0, [1, 2, 3, 4, 0, 0, 0, 1, 1], 0, [1, 2, 3, 4, 0, 1, 0, 1, 1]]
...(ten different archs)
arch10 = [...]
"""

movielens_prompt_nd = """
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.
The downstream task is link prediction, and the target edge is N2 - N4. 

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.
And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min'].After selecting all operation types, You just need to output the index corresponding to that operation in the search space.

When considering the two-layer architecture, the message delivery path is as follows:
layer_0': {N1-N1: '0:zero', N1-N2: '0:zero', N3-N4: '1:gcn', N3-N3: '0:zero', N4-N4(self): '1:gcn', N4-N3: '1:gcn', 
N4-N4: '1:gcn', N4-N2: '1:gcn', N5-N2: '1:gcn', N5-N5: '1:gcn'', N2-N1: '1:gcn', N2-N4: '1:gcn', N2-N5: '1:gcn', N2-N2: '1:gcn', N2-N2(self): '1:gcn', 'multi_aggr': '0"sum'}, 
'layer_1': {N1-N2: '1:gcn', N3-N4: '1:gcn', N4-N4(self): '1:gcn', N4-N4: '1:gcn', N4-N2: '1:gcn', N5-N2: '0:zero', N2-N4: '0:zero', N2-N2: '1:gcn', N2-N2(self): '1:gcn', 'multi_aggr': '0"sum'}, 
'inter_layer': {N1-N2: '1:gcn', N3-N4: '1:gcn', N4-N4(self): '1:gcn', N4-N4: '1:gcn', N4-N2: '1:gcn', N5-N2: '1:gcn', N2-N4: '0:zero', N2-N2: '1:gcn', N2-N2(self): '1:gcn'}}
'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.
According to the search space requirements, this architecture is expressed as [[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 0, [1, 1, 1, 1, 1, 0, 0, 1, 1], 0, [1, 1, 1, 1, 1, 1, 0, 1, 1]]

#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Avoid generating existing architectures
here is the output format:
arch1 = [[0, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 0, 1, 0, 1], 4, [4, 3, 2, 1, 0, 4, 3, 2, 1], 4, [4, 3, 2, 1, 0, 4, 3, 2, 1]]
arch2 = [[1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0], 0, [1, 2, 3, 4, 0, 0, 0, 1, 1], 0, [1, 2, 3, 4, 0, 1, 0, 1, 1]]
...(ten different archs)
arch10 = [...]
"""

yelp_prompt="""
task description：
Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.
You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.
dataset:
Yelp, User(U) node:16,240, Business(B) node:14285, Category(C) node :512, City(I):48,Compliment(O) node:12.U-B edge:67008, B-C edge:40009,B-I edge 14267,U-O edge:76875,U-U edge:158590.

The downstream task is link prediction,and the target edge is U-B. When considering the three layer architecture, the message delivery path is as follows:
search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge
And multi_aggr should choose from['0:sum].After selecting all operation types, You just need to output the index corresponding to that operation in the search space.

When considering the two-layer architecture, the message delivery path is as follows:

'layer_0': {B-B: '0:zero', B-C: '0:zero', B-I: '0:zero', B-U: '1:gcn', C-B: '1:gcn', C-C: '0:zero', I-I: '1:gcn', I-B: '0:zero', O-O: '1:gcn', O-U: '1:gcn', U-B: '0:zero', U-O: '0:zero', U-U: '0:zero', U-U(self): '1:gcn', 'multi_aggr': '0:sum'}, 
'layer_1': {B-B: '1:gcn', B-U: '1:gcn', C-B: '0:zero', I-B: '0:zero', O-U: '0:zero', U-B: '0:zero', U-U: '1:gcn', U-U(self): '0:zero', 'multi_aggr': '0:sum'}, 
'inter_layer': {B-B: '0:zero', B-U: '1:gcn', C-B: '0:zero', I-B: '0:zero', O-U: '1:gcn', U-B: '0:zero', U-U: '0:zero', U-U(self): '1:gcn'}}
Where U-B represents the source node business, the target node user, and their edge is U-B. 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.
According to the search space requirements, this architecture is expressed as [[0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1], 0, [1, 1, 0, 0, 0, 0, 1, 0], 0,[ 0, 1, 0, 0, 1, 0, 0, 1]]
#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:
arch1 = [[4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 4, 3, 2, 1], 0, [4, 3, 2, 1, 0, 4, 3, 2], 0, [4, 3, 2, 1, 0, 4, 3, 2]]
arch2 = [[0, 0, 0, 0, 0, 0, 4, 3, 2, 1, 0, 4, 3, 2], 0, [2, 2, 2, 2, 1, 1, 1, 1], 0, [3, 3, 3, 3, 1, 1, 1, 1]]
...(ten different archs)
arch10 = [...]
"""

amazon_prompt="""
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.

Amazon, User(U) node 6170, Item(I) node 2753, View(V) ndoe 3857, Brand(B) node 334,Category(C) node 22.U-I edge 195,791, I-V edge 5694,I-C edge 5508,I-B edge 2753

The downstream task is link prediction, and the target edge is U-I.

search space：
For each edge, we need to select one from [0:zero,1:gcn,2:gat,3:edge,4:sage], zero means that the message passing of this type of edge is not considered, gcn means that the GCN graph convolutional neural network is used to process neighbor information for this type of edge, The other three are also powerful graph convolutional operators.
And multi_aggr should choose one from['0:sum', '1:max', '2:mean', "3:att", '4:min'].After selecting all operation types, You just need to output the index corresponding to that operation in the search space.

When considering the two-layer architecture, the message delivery path is as follows:
layer_0: {B-B: '3:edge', B-I: '1:gcn', C-I: '1:gcn', C-C: '0:zero', I-I: '2:gat', I-B: '3:edge', I-C: '0:zero', I-U: '3:edge', I-V: '4:sage', U-I: '1:gcn', U-U: '0:zero', V-I: '2:gat', V-V: '2:gat'} 'multi_aggr': '0:sum', 
layer_1: {B-I: '2:gat', C-I: '0:zero', I-I: '2:gat', I-U: '4:sage', U-I: '4:sage', U-U: '3:edge', V-I: '4:sage'}, 'multi_aggr': 'sum', 
inter_modes': {B-I: '1:gcn', C-I: '0:zero', I-I: '4:sage', I-U: '4:sage', U-I: '1:gcn', U-U: '3:edge', V-I: '3:edge'}}
Where U-B represents the source node business, the target node user, and their edge is U-B. 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.
According to the search space requirements, this architecture is expressed as [[3, 1, 1, 0, 2, 3, 0, 3, 4, 1, 0, 2, 2], 0, [2, 0, 2, 4, 4, 3, 4], 0, [1, 0, 4, 4, 1, 3, 3]]
#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:
arch1 = [[3, 0, 4, 4, 2, 3, 2, 1, 1, 2, 4, 2, 2], 3, [1, 1, 2, 2, 2, 4, 0], 3, [0, 3, 1, 3, 2, 1, 1]]
arch2 = [[3, 1, 1, 0, 2, 3, 0, 3, 4, 1, 0, 2, 2], 0, [2, 0, 2, 4, 4, 3, 4], 0, [1, 0, 4, 4, 1, 3, 3]]
...(ten different archs)
arch10 = [...]
"""

amazon_prompt_ns="""
task description：

Your task is to perform neural network architecture search, that is, to find the most suitable neural network structure for a given data set from a given search space.

You need to sample possible neural network architectures from the search space as the resulting output. You will receive the accuracy rate of the sampled architecture on the data set. You need to self-learn and adjust the search strategy according to the accuracy rate, and finally give the optimal neural network structure to make the accuracy rate of downstream tasks (such as recommendation system, node classification) Highest.

Amazon, User(U) node 6170, Item(I) node 2753, View(V) ndoe 3857, Brand(B) node 334,Category(C) node 22.U-I edge 195,791, I-V edge 5694,I-C edge 5508,I-B edge 2753

The downstream task is link prediction, and the target edge is U-I.

search space：
For each edge, we need to select one from [0,1,2,3,4]
And multi_aggr should choose one from[0, 1, 2, 3, 4].After selecting all operation types, You just need to output the index corresponding to that operation in the search space.

When considering the two-layer architecture, the message delivery path is as follows:
layer_0: {B-B: '3', B-I: '1', C-I: '1', C-C: '0', I-I: '2', I-B: '3', I-C: '0', I-U: '3', I-V: '4', U-I: '1', U-U: '0', V-I: '2', V-V: '2'} 'multi_aggr': '0', 
layer_1: {B-I: '2', C-I: '0', I-I: '2', I-U: '4', U-I: '4', U-U: '3', V-I: '4'}, 'multi_aggr': '0', 
inter_modes': {B-I: '1', C-I: '0', I-I: '4', I-U: '4', U-I: '1', U-U: '3', V-I: '3'}}
Where U-B represents the source node business, the target node user, and their edge is U-B. 'multi_aggr' represents the message aggregation function.In addition,'inter_layer' represents cross-layer links.
According to the search space requirements, this architecture is expressed as [[3, 1, 1, 0, 2, 3, 0, 3, 4, 1, 0, 2, 2], 0, [2, 0, 2, 4, 4, 3, 4], 0, [1, 0, 4, 4, 1, 3, 3]]
#Remember, Architecture length should be consistent and no options should be chosen that are not available in the search space#
You should generate ten different architectures,which must fit within the scope of the search space and The length of the schema needs to remain the same as in the example.Do not generate the architectures provided in the past.Avoid generating existing architectures
here is the output format:
arch1 = [[3, 0, 4, 4, 2, 3, 2, 1, 1, 2, 4, 2, 2], 3, [1, 1, 2, 2, 2, 4, 0], 3, [0, 3, 1, 3, 2, 1, 1]]
arch2 = [[3, 1, 1, 0, 2, 3, 0, 3, 4, 1, 0, 2, 2], 0, [2, 0, 2, 4, 4, 3, 4], 0, [1, 0, 4, 4, 1, 3, 3]]
...(ten different archs)
arch10 = [...]
"""

from hgnn.configs import register_hgnn_args
from hgnn.meta_manager import AggrManagerSK
from hgnn.utils import set_seed
from nas.search_space import SearchSpace
from benchmark.random_build_bench import HGNNRecord
import argparse
import heapq
import json
import requests
import time

import numpy as np

from http import HTTPStatus
import dashscope


def main():
    messages = [{'role': 'system', 'content': 'You are a neural network architecture search controller, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users.Please check the feedback provided by users, and your newly generated architecture should be different from the existing architecture.'},
                {'role': 'user', 'content':   amazon_prompt},]
    for i in range(15):
        response = dashscope.Generation.call(
            model='llama2-13b-chat-v2',
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            print(response["output"]["choices"][0]["message"]["content"])
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
        try:
            lines = response["output"]["choices"][0]["message"]["content"].split('\n')

            for line in lines:
                if not line.strip():
                        continue

                            
                if '[' in line and ']' in line:
                    start = line.find('[')
                    end = line.find(']')
                    architecture = line[start:end+1]
                    print(architecture)
                # action_name, action_content = line.split('=')
                try:    
                    code = eval(architecture)
                    record = HGNNRecord()
                    record.code = code
                    if len(code) != code_len or max(code)>=5:continue
                    if code in code_list: continue
                    record.desc = search_space.decode(code)  # {'layer_0': OrderedDict([(('author', 'aa', 'author'), '1:gcn'), (('author', 'ap', 'paper'), '1:gcn'), (('field', 'ff', 'field'), '0:zero'), (('field', 'fp', 'paper'), '0:zero'), (('paper', 'pa', 'author'), '1:gcn'), (('paper', 'pf', 'field'), '1:gcn'), (('paper', 'pp', 'paper'), '1:gcn'), ('multi_aggr', '0:sum')]), 'layer_1': OrderedDict([(('author', 'ap', 'paper'), '0:zero'), (('field', 'fp', 'paper'), '1:gcn'), (('paper', 'pp', 'paper'), '0:zero'), ('multi_aggr', 'min')]
                # run and calculate time
                    set_seed(1)
                    val_score, test_score = gnn_manager_obj.evaluate(record.desc)
                    code_list.append(code)
                    val_list.append(val_score)
                    test_list.append(test_score)
                    performance_history.append({"code": code, "val_score": val_score, "test_score": test_score})
                except:
                    continue
            sorted_performance = sorted(performance_history, key=lambda x: x['val_score'], reverse=True)
            if i < 10:
                prompt_last = '''In the previous round of experiments, the architectures you provided me and their corresponding performance are as follows:\n{}''' \
                    .format(''.join(
                    ['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in performance_history[-50:]]))
                #prompt_last += "Now your task is to analyze how to obtain a better architecture based on the architecture and accuracy provided previously, use the analysis results and search strategy and design ten new different new architectures."
                prompt_last += "search strategy：You should look for architectures that have not been explored before." 
                prompt_last += "Now your task is to design ten new different new architectures and try to find better architectures"
            else:
                top_20 = sorted_performance[:20]

                latest_20 = performance_history[-20:]

                prompt_last = "In the previous round of experiments, the architectures you provided me and their corresponding performance are as follows:\n"
        
                prompt_last += "\nLatest architectures you provided:\n"
                prompt_last += ''.join(['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in latest_20])

                prompt_last += "Top 20 architectures by validation accuracy:\n"
                prompt_last += ''.join(['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in top_20])

                prompt_last += '"search strategy:Now that you have accrued the accuracy of some architectures, Analyze how to get better architectures based on the characteristics of well-performing architectures and continue sampling architectures. Please design ten new architectures based on your analysis'
        
            messages = [{'role': 'system', 'content': 'You are a neural network architecture search controller, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. Please note to respond in the format specified by the user.Do not output other content except generating architectures.'},
                    {'role': 'user', 'content':  dblp_prompt + prompt_last}]
            print(messages)
            messages_history.append(messages)
        except:
            continue
    with open("/history/llama/amazon_llama_message.json", 'w') as f:
         json.dump(messages_history, f)
    with open("/history/llama/amazon_llama_performance.json", 'w') as f:
        json.dump(performance_history ,f)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HGNAS')
    register_hgnn_args(parser)
    args = parser.parse_args()
    # args.dataset = 'dblp'
    # gnn_manager_obj = AggrManagerSK(args)
    # edge_dict = gnn_manager_obj.edge_dict
    # predict_keys = gnn_manager_obj.predict_keys
    # search_space = SearchSpace(
    #     edge_types=edge_dict, n_layers=2, predict_keys=predict_keys,
    #     full_gnn_list=True, contain_zero=not False, predict_inter_layer=True
    # )
    # code_len = len(search_space.sample(1)[0])
    messages_history = []
    performance_history = []
    code_list = []
    val_list = []
    test_list = []
    from final_archs.general_manager import GeneralManagerLP
    from link_predict.meta_manager import MetaOptLinkPredictorManager
    import torch
    args.dataset = "amazon"
    args.task = "lp"
    args.use_feat = False
    args.metrics = "auc"
    set_seed(args.random_seed)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    manager = GeneralManagerLP(args, device)
    gnn_manager_obj = MetaOptLinkPredictorManager(args)

    edge_dict = gnn_manager_obj.edge_dict
    predict_keys = [gnn_manager_obj.pre_dst_type,gnn_manager_obj.pre_src_type,gnn_manager_obj.pre_link]

    search_space = SearchSpace(
        edge_types=edge_dict, n_layers=2, predict_keys=predict_keys,
        full_gnn_list=True, contain_zero=not False, predict_inter_layer=True
    )
    code_len = len(search_space.sample(1)[0])
    sorted_performance = sorted(performance_history, key=lambda x: x['val_score'], reverse=True)
                   # run and calculate time
    set_seed(1)
    main()
