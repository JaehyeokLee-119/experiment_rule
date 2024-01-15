'''
jsonl 데이터를 불러오는 dataloader



'''
from datasets import load_dataset
from typing import List

def answers_to_string(answers: List[str]) -> str:
    answer_str = ""
    for a, answer in enumerate(answers):
        if a == len(answers) - 1:
            answer_str += f"{a+1}: {answer}"
        else:
            answer_str += f"{a+1}: {answer}, "
    return answer_str

def template_to_string(template: str, context: str, question: str, options: List[str]) -> str:
    # option combinate
    option_str = answers_to_string(options)
    return template.format(context=context, question=question, options=option_str)

def dataload_jsonl_reclor(data_path: str, portion: str = "100%"):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, split=f"train[:{portion}]")
    else:
        data = load_dataset(data_path, split=f"train[:{portion}]")
    return data

def extract_response_from_output(output_string: str):
    # string에서 등장하는 마지막 '### Response:' 이후의 string을 반환
    key = '[\INST]' # '### Response:'
    # output_string = output_string[output_string.rfind('### Response: ')+len('### Response: '):]
    output_string = output_string[output_string.rfind(key)+len(key):]
    if '</s>' in output_string: # 없애기
        output_string = output_string[:output_string.find('</s>')]
    return output_string.strip()