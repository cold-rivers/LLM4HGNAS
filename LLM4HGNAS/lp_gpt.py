
from hgnn.configs import register_hgnn_args
from hgnn.meta_manager import AggrManagerSK
from hgnn.utils import set_seed
from nas.search_space import SearchSpace
import argparse
import sys
import heapq
import json
import requests
import time
import numpy as np
from prompt import douban_prompt,movielens_prompt,yelp_prompt,amazon_prompt
import torch
url = "https://api.openai.com/v1/chat/completions"

system_content='''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.Do not generate the architectures provided in the past'''
performance_history=[]


def experiments_prompt(code_list, value_list):
    prompt_last = '''In the previous round of experiments, the models you provided me and their corresponding performance are as follows:\n{}''' \
        .format(''.join(
        ['arch {} achieves accuracy {:.4f} .\n'.format(code, val_score) for code, val_score in
         zip(code_list, value_list)]))
    return prompt_last


headers = {
    "Authorization": "Bearer " + "your api key",
    "Content-Type": "application/json"
}

def flatten_list(code):
    flat_list = []
    for element in code:
        if isinstance(element, list):
            flat_list.extend(element)
        else:
            flat_list.append(element)
    return flat_list

def gpt_hgnas():
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": main_prompt},
    ]
    print(messages)

    llm_time = 0
    for i in range(15):
        da = {
            "model": "gpt-4-0314",#gpt-4-0314"
            "messages": messages,
            "temperature": 1
        }
        try:
            post_time = time.time()
            response = requests.post(url, headers=headers, data=json.dumps(da))
            response_time = time.time()
            llm_time += response_time - post_time
            print("response time:", response_time - post_time)

            res = response.json()
            print(res)
            res_temp = res['choices'][0]['message']['content']

            print(res_temp)
            messages.append({"role": "assistant", "content": res_temp})
            messages_history.append(messages)
            try:
                lines = res_temp.split('\n')

                for line in lines:
                        if not line.strip():
                                continue

                        if '[[' in line and ']]' in line:
                            start = line.find('[[')
                            end = line.find(']]')
                            architecture = line[start:end+2]
                            print(architecture)
                        # action_name, action_content = line.split('=')
                        try:    
                            ori_code = eval(architecture)
                            # for j in range(len(ori_code)):
                            #     if j == 1 or j == 3: continue
                            #     for k in range(len(ori_code[j])):
                            #         if ori_code[j][k]>1:ori_code[j][k]=1
                            code = flatten_list(ori_code)
                            print(code)
                            if len(code) != code_len or max(code)>=5:
                                print("error arch")
                                continue
                            print(code)
                            if code not in code_list: 
                                desc = search_space.decode(code)  
                                set_seed(1)
                                start_time = time.time()
                                val_score, test_score = gnn_manager_obj.evaluate(desc)
                                end_time = time.time()
                                print("val_score:", val_score, "test_score:", test_score)
                                time_cost = end_time - start_time
                                code_list.append(code)
                                val_list.append(val_score)
                                test_list.append(test_score)
                                performance_history.append({"code": ori_code, "val_score": val_score, "test_score": test_score,"time":time_cost})
                            else:
                                code_list.append(code)
                                index = code_list.index(code)
                                performance_history.append(performance_history[index])
                        except Exception as e:
                            print(e)
                            continue
            except Exception as e:
                print(e)
                continue
        except Exception as e:
            print(e)
            with open(f"{dataset}_message.json", 'w') as f:
                json.dump(messages_history, f)
            with open(f"{dataset}_performance.json", 'w') as f:
                json.dump(performance_history ,f)
            continue
        sorted_performance = sorted(performance_history, key=lambda x: x['val_score'], reverse=True)
        if i < 15:
            prompt_last = '''In the previous round of experiments, the architectures you provided me and their corresponding performance are as follows:\n{}''' \
                .format(''.join(
                ['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in performance_history[-30:]]))
            prompt_last += "search strategy:you should look for architectures that have not been explored before."
            prompt_last += "please use search strategy to design ten new different architectures and try to get better architectures."
        else:
            top = sorted_performance[:20]
            latest = performance_history[-20:]
            prompt_last = "In the previous round of experiments, the architectures you provided me and their corresponding performance are as follows:\n"

            prompt_last += "\nLatest architectures you provided:\n"
            prompt_last += ''.join(['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in latest])

            prompt_last += "Top architectures by validation accuracy:\n"
            prompt_last += ''.join(['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in top])

            prompt_last += 'search strategy:Now that you have accrued the accuracy of some architectures, Analyze how to get better architectures based on the well-performing architectures. Please give the analysis results and sampling basis and design ten new architectures based on your analysis and search strategy'
        messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": main_prompt + prompt_last}]
        #messages.append({"role": "user", "content":prompt_last + "please give 10 other architectures and try to find better architectures.#Avoid generating architectures provided in previous rounds#"})
        print(messages)
        messages_history.append(messages)
    performance_history.append({"llm_time":llm_time})
    with open(f"{dataset}_message.json", 'w') as f:
         json.dump(messages_history, f)
    with open(f"{dataset}_performance.json", 'w') as f:
        json.dump(performance_history ,f)


if __name__ == '__main__':
    messages_history = []
    performance_history = []
    code_list = []
    test_list = []
    val_list = []
    code_list = []
    parser = argparse.ArgumentParser('LLM4HGNAS')
    register_hgnn_args(parser)
    args = parser.parse_args()

    from final_archs.general_manager import GeneralManagerLP
    from link_predict.meta_manager import MetaOptLinkPredictorManager
    datasets = ["movielens", "amazon", "yelp"]
    dataset = args.dataset.lower()
    if dataset not in datasets:
        print(f"Dataset '{args.dataset}' is not an allowed dataset. Please choose from {datasets}.")
        sys.exit(1)
    main_prompt = ""    
    if dataset == 'amazon':main_prompt = amazon_prompt
    elif dataset == 'yelp':main_prompt = yelp_prompt
    else: main_prompt == movielens_prompt

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
    gpt_hgnas()
