
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
import sys
from prompt import dblp_prompt,imdb_prompt,acm_prompt
url = "https://api.openai.com/v1/chat/completions"

system_content='''You are a neural network architecture search AI, and you need to provide the architecture that users need and think about how to obtain a better performing architecture based on the architecture performance feedback from users. You need to gradually solve the problem. Please note to respond in the format specified by the user.Do not generate the architectures provided in the past'''#这里放入系统级提示
performance_history=[]


def experiments_prompt(code_list, value_list):
    prompt_last = '''In the previous round of experiments, the models you provided me and their corresponding performance are as follows:\n{}''' \
        .format(''.join(
        ['arch {} achieves accuracy {:.4f} .\n'.format(code, val_score) for code, val_score in
         zip(code_list, value_list)]))
    return prompt_last


headers = {
    "Authorization": "Bearer " + "your key",
    "Content-Type": "application/json"
}
def gpt_hgnas():
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": main_prompt},
    ]
    print(messages)

    
    import time
    llm_time = 0
    for i in range(15):
        da = {
            "model": "gpt-4-0314",
            "messages": messages,
            "temperature": 1
        }
        try:
            post_time = time.time()
            response = requests.post(url, headers=headers, data=json.dumps(da))
            response_time = time.time()
            llm_time += response_time - post_time
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

                        if '[' in line and ']' in line:
                            start = line.find('[')
                            end = line.find(']')
                            architecture = line[start:end+1]
                            print(architecture)
                        try:    
                            code = eval(architecture)
                            record = HGNNRecord()
                            record.code = code
                            if len(code) != code_len or max(code)>=5:continue
                            # for i in range(15):
                            #     if i == 7 or i == 11: continue
                            #     if code[i]>1:code[i]=1
                            #if code in code_list: continue
                            record.desc = search_space.decode(code)  
                        # run and calculate time
                            set_seed(1)
                            start_time = time.time()
                            val_score, test_score = gnn_manager_obj.evaluate(record.desc)
                            end_time = time.time()
                            code_list.append(code)
                            val_list.append(val_score)
                            test_list.append(test_score)
                            performance_history.append({"code": code, "val_score": val_score, "test_score": test_score,"time":end_time-start_time})
                        except:
                            continue
            except:
                continue
        except:
            with open(f"'{dataset}'_message.json", 'w') as f:
                json.dump(messages_history, f)
            with open(f"'{dataset}'_performance.json", 'w') as f:
                json.dump(performance_history ,f)
            continue
        sorted_performance = sorted(performance_history, key=lambda x: x['val_score'], reverse=True)
        if i < 10:
            prompt_last = '''In the previous round of experiments, the architectures you provided me and their corresponding performance are as follows:\n{}''' \
                .format(''.join(
                ['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in performance_history[-50:]]))
            #prompt_last += "Now your task is to analyze how to obtain a better architecture based on the architecture and accuracy provided previously, use the analysis results and search strategy and design ten new different new architectures."
            prompt_last += "search strategy:you should look for architectures that have not been explored before." 
            prompt_last += "Now your task is to design ten new different new architectures and try to find better architectures"
        else:
            top = sorted_performance[:20]

            latest = performance_history[-30:]

            prompt_last = "In the previous round of experiments, the architectures you provided me and their corresponding performance are as follows:\n"
       
            prompt_last += "\nLatest architectures you provided:\n"
            prompt_last += ''.join(['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in latest])

            prompt_last += "Top 20 architectures by validation accuracy:\n"
            prompt_last += ''.join(['arch {} accuracy {:.4f} .\n'.format(item['code'], item['val_score']) for item in top])

            prompt_last += '"search strategy:Now that you have accrued the accuracy of some architectures, Analyze how to get better architectures based on the characteristics of well-performing architectures and continue sampling architectures. Please design ten new architectures based on your analysis'
        messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": main_prompt + prompt_last }]
        #messages.append({"role": "user", "content":prompt_last + "please give 10 other architectures and try to find better architectures.#Avoid generating architectures provided in previous rounds#"})
        print(messages)
        messages_history.append(messages)
    messages_history.append({"llm_time":llm_time})
    with open(f"'{dataset}'_message.json", 'w') as f:
         json.dump(messages_history, f)
    with open(f"'{dataset}'_performance.json", 'w') as f:
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
    datasets = ["acm", "dblp", "imdb"]
    dataset = args.dataset.lower()
    if dataset not in datasets:
        print(f"Dataset '{args.dataset}' is not an allowed dataset. Please choose from {datasets}.")
        sys.exit(1)
    main_prompt = ""    
    if dataset == 'acm':main_prompt = acm_prompt
    elif dataset == 'dblp':main_prompt = dblp_prompt
    else: main_prompt == imdb_prompt
    gnn_manager_obj = AggrManagerSK(args)
    edge_dict = gnn_manager_obj.edge_dict
    predict_keys = gnn_manager_obj.predict_keys
    search_space = SearchSpace(
        edge_types=edge_dict, n_layers=2, predict_keys=predict_keys,
        full_gnn_list=True, contain_zero=not False, predict_inter_layer=True
    )
    code_len = len(search_space.sample(1)[0])
    #sorted_performance = sorted(performance_history, key=lambda x: x['val_score'], reverse=True)
        
    #print(args)
    gpt_hgnas()
