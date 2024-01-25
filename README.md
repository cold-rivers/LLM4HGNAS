# Heterogeneous Graph Neural Architecture Search with Large Language Models
code for https://arxiv.org/abs/2312.08680
## Installation
```
git clone https://github.com/cold-rivers/LLM4HGNAS.git
cd LLM4HGNAS
pip install -r requirements.txt
```

## Usage

1. Unzip ```data/data.zip```, put Amazon, DBLP, IMDB, Movielens, Yelp under ```data/```

2. Set your openai key in ```LLM4HGNAS/nc_gpt.py``` and ```LLM4HGNAS/lp_gpt.py```
```
headers = {
    "Authorization": "Bearer " + "your api key",
    "Content-Type": "application/json"
}
```

3  Search
```
cd LLM4HGNAS
# node classification
python nc_gpt.py --dataset ACM
# link prediction
python lp_gpt.py --dataset Amazon
```
response of GPT
```
#ACM
{[{'index': 0, 'message': {'role': 'assistant', 'content': '
arch1=[1,2,3,4,0,1,2,3,4,1,0,1,2,3,4]\n\n
arch2=[0,1,2,3,4,1,2,3,4,0,2,3,4,1,0]\n\n
arch3=[2,3,4,0,1,2,3,4,0,1,3,4,1,2,0]\n\n
arch4=[4,0,1,2,3,4,0,1,2,3,4,0,1,2,3]\n\n
arch5=[3,2,1,0,4,3,2,1,0,4,3,2,1,0,4]\n\n
arch6=[1,4,3,2,0,1,4,3,2,0,1,4,3,2,0]\n\n
arch7=[2,0,4,3,1,2,0,4,3,1,2,0,4,3,1]\n\n
arch8=[0,3,1,2,4,0,3,1,2,4,0,3,1,2,4]\n\n
arch9=[3,2,0,1,4,3,2,0,1,4,3,2,0,1,4]\n\n
arch10=[1,3,4,2,0,1,3,4,2,0,1,3,4,2,0]'},
'logprobs': None, 'finish_reason': 'stop'}]}

#Amazon
 [{'index': 0, 'message': {'role': 'assistant', 'content': '
arch1 = [[2, 0, 3, 2, 4, 4, 1, 2, 1, 3, 4, 0, 3], 2, [1, 1, 3, 2, 2, 0, 4], 1, [4, 0, 2, 1, 2, 3, 3]]\n
arch2 = [[1, 2, 1, 3, 0, 3, 4, 2, 3, 1, 2, 4, 3], 0, [3, 4, 2, 2, 1, 0, 3], 4, [1, 3, 0, 2, 1, 2, 3]]\n
arch3 = [[3, 1, 2, 4, 0, 3, 2, 2, 1, 1, 3, 3, 4], 2, [2, 3, 4, 1, 1, 0, 4], 0, [3, 1, 2, 0, 4, 2, 3]]\n
arch4 = [[0, 4, 2, 3, 1, 2, 3, 2, 0, 3, 1, 4, 4], 4, [3, 0, 3, 1, 2, 4, 2], 1, [2, 3, 1, 0, 4, 2, 3]]\n
arch5 = [[3, 4, 4, 1, 0, 2, 2, 3, 1, 2, 3, 0, 1], 3, [1, 0, 4, 2, 3, 2, 1], 4, [2, 0, 1, 3, 2, 3, 4]]\n
arch6 = [[1, 2, 0, 3, 2, 1, 4, 3, 4, 0, 2, 3, 4], 0, [3, 4, 1, 2, 3, 0, 2], 2, [1, 3, 2, 4, 0, 1, 3]]\n
arch7 = [[3, 0, 2, 1, 4, 3, 4, 1, 2, 3, 0, 2, 3], 3, [1, 4, 2, 3, 0, 2, 1], 1, [2, 3, 1, 0, 4, 2, 4]]\n
arch8 = [[2, 3, 1, 0, 2, 4, 3, 2, 1, 4, 0, 1, 3], 2, [4, 0, 2, 1, 3, 2, 1], 0, [3, 1, 2, 0, 4, 3, 4]]\n
arch9 = [[2, 3, 1, 0, 2, 4, 1, 3, 4, 0, 2, 4, 3], 0, [3, 2, 1, 4, 3, 0, 2], 4, [1, 3, 2, 4, 0, 2, 3]]\n
arch10 = [[0, 2, 3, 4, 1, 2, 3, 1, 0, 2, 1, 4, 3], 4, [2, 3, 1, 2, 1, 4, 3], 2, [1, 3, 2, 0, 4, 2, 3]]'}
, 'logprobs': None, 'finish_reason': 'stop'}]
```

4 Evaluate
```
cd final_archs
python acm.py
```


## Citation
```
@misc{dong2023heterogeneous,
      title={Heterogeneous Graph Neural Architecture Search with GPT-4}, 
      author={Haoyuan Dong and Yang Gao and Haishuai Wang and Hong Yang and Peng Zhang},
      year={2023},
      eprint={2312.08680},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
