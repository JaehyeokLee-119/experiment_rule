import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import List
import time
import os 
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import pickle
import json 
load_dotenv()
from GPT_request import ask_chatgpt
from tqdm import tqdm

API_KEY = os.getenv("API_KEY")

TEMPLATE = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction: You are given a logical reasoning question and predefined types of questions. Each type has its unique type code such as 'N1'. write the code of the question type of following question.
There are predefined 17 types of questions. The number of the question type and the description of the question type are as follows: 
N1. "Necessary Assumptions": identify the claim that must be true or is required in order for the argument to work.
N2. "Sufficient Assumptions": identify a sufficient assumption, that is, an assumption that, if added to the argument, would make it logically valid.
N3. "Strengthen": identify information that would strengthen an argument
N4. "Weaken": identify information that would weaken an argument
N5. "Evaluation": identify information that would be useful to know to evaluate an argument
N6. "Implication": identify something that follows logically from a set of premises
N7. "Conclusion/Main Point": identify the conclusion/main point of a line of reasoning
N8. "Most Strongly Supported": find the choice that is most strongly supported by a stimulus
N9. "Explain or Resolve": identify information that would explain or resolve a situation
N10. "Principle": identify the principle, or find a situation that conforms to a principle, or match the principles
N11. "Dispute": identify or infer an issue in dispute
N12. "Technique": identify the technique used in the reasoning of an argument
N13. "Role": describe the individual role that a statement is playing in a larger argument
N14. "Identify a Flaw": identify a flaw in an argument’s reasoning
N15. "Match Flaws": find a choice containing an argument that exhibits the same flaws as the passage’s argument
N16. "Match Structures": match the structure of an argument in a choice to the structure of the argument in the passage
N17. "Others": other types of questions which are not included by the above

Following logical reasoning question.
Context: {context}
Question: {question}
Options: {options}
The code of the question type of this question is N
""".strip()

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


class GPT_run:
    def __init__(self, model_name, file_name, outputfile_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=API_KEY)
        self.template = TEMPLATE 
        self.filename = file_name
        
    def run(self):
        for ix, x in enumerate(data_loader):
            context = x["context"][0] # str
            question = x["question"][0]
            answers = x["answers"] # List 
            answers = [x[0] for x in answers]
            question_type = x["question_type"][0] # int
            id_string = x["id_string"][0] # str
            
            input_string = template_to_string(self.template, context, question, answers)
            

if __name__ == '__main__':
    file_name = '../../Data/ReClor/ReClor_test.jsonl'
    outputfile_name = 'qtype_result/ReClor_test_result.jsonl'
    data_dataset = load_dataset("json", data_files=file_name, split="train")
    data_loader = DataLoader(data_dataset, batch_size=1, shuffle=False)
    
    model_name = 'gpt-4-1106-preview'
    gpt = GPT_run(model_name, file_name, outputfile_name)
    gpt.run()
