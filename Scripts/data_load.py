'''
jsonl 데이터를 불러오는 dataloader



'''
from datasets import load_dataset
from typing import List

def answers_to_string(answers: List) -> str:
    answer_str = ""
    for a, answer in enumerate(answers):
        while type(answer) == tuple:
            answer = answer[0]
        if a == len(answers) - 1:
            answer_str += f"{a+1}: {answer}"
        else:
            answer_str += f"{a+1}: {answer}, "
    return answer_str

def template_to_string(template: str, context: str, question: str, options: List[str]) -> str:
    # option combinate
    option_str = answers_to_string(options)
    return template.format(context=context, question=question, options=option_str)

def template_to_string_with_mainq(template: str, context: str, question: str, options: List[str], mainq_question: str, mainq_options: List[str], main_label: int = 1) -> str:
    # option combinate
    option_str = answers_to_string(options)
    mainq_option_str = answers_to_string(mainq_options)
    
    # template에 {mainq_label}이 있는지 확인
    if '{mainq_label}' in template:
        # mainq_label을 추가
        mainq_label = main_label+1
        return template.format(context=context, question=question, options=option_str, main_question=mainq_question, main_options=mainq_option_str, mainq_label=mainq_label)
    else:
        return template.format(context=context, question=question, options=option_str, mainq_question=mainq_question, mainq_options=mainq_option_str)

def dataload_jsonl_reclor(data_path: str, portion: str = "100%"):
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path, split=f"train[:{portion}]")
    else:
        data = load_dataset(data_path, split=f"train[:{portion}]")
    return data

def extract_response_from_output(output_string: str):
    # string에서 등장하는 마지막 '### Response:' 이후의 string을 반환
    if '[\INST]' in output_string:
        key = '[\INST]'
    elif '### Response:' in output_string:
        key = '### Response:'
    # output_string = output_string[output_string.rfind('### Response: ')+len('### Response: '):]
    output_string = output_string[output_string.rfind(key)+len(key):]
    if '</s>' in output_string: # 없애기
        output_string = output_string[:output_string.find('</s>')]
    return output_string.strip()