import os 
import transformers 
import fire 
# CUDA_VISIBLE_DEVICES=3 python download.py

os.environ["TRANSFORMERS_CACHE"] = "/hdd/hjl8708/saved_models"

import torch 
from torch.utils.data import DataLoader, DistributedSampler
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from Scripts.data_load import dataload_jsonl_reclor, template_to_string, extract_response_from_output
from peft import (
    LoraConfig,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PeftModel,
)
from tqdm import tqdm 
import torch.distributed as dist

# 명령어


def load_model_and_tokenizer(
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    reduce_memory: str = "4-bit", # "None", half-precision, "4-bit", "8-bit"
    flash_attention: bool = False,
    device_map: str = "auto",    
): 
    # model load (model_name, reduce_memory, flash_attention)
    if reduce_memory == "4-bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir='/hdd/hjl8708/saved_models', 
            load_in_4bit=True,
            use_flash_attention_2=flash_attention,
            device_map=device_map)
    elif reduce_memory == "8-bit":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir='/hdd/hjl8708/saved_models', 
            load_in_8bit=True,
            use_flash_attention_2=flash_attention,
            device_map=device_map)
    elif reduce_memory == "half-precision":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir='/hdd/hjl8708/saved_models', 
            load_in_half_precision=True,
            use_flash_attention_2=flash_attention,
            device_map=device_map)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir='/hdd/hjl8708/saved_models', 
            use_flash_attention_2=flash_attention,
            device_map=device_map)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
    
def inference(
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    template_path: str = "/hdd/hjl8708/workspace/Inference/templates/mixtral.json",
    template_key: str = "prompt_inference_onlynum",
    peft_model_path: str = "",
    data_path: str = "/hdd/hjl8708/workspace/Data/RULE/RULE_subq_all.jsonl",
    output_path: str = "/hdd/hjl8708/workspace/Inference/Result",
    output_key: str = "RULE_subq_Mixtral_4bit",
    reduce_memory: str = "4-bit", # "None", half-precision, "4-bit", "8-bit"
    flash_attention: bool = False,
):
    world_size = int(os.environ.get("WORLD_SIZE", 1)) # ddp 프로세스 개수
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) # ddp 사용 시 현재 프로세스 번호
    ddp = world_size != 1 # ddp를 사용할지 여부 
    if ddp: # ddp 사용 시의 설정
        device_map = {"": local_rank} # 프로세스의 gpu 번호
        print(f"Using DDP with local rank {local_rank}")
    else:
        device_map = "auto"
        
    model, tokenizer = load_model_and_tokenizer(model_name, reduce_memory, flash_attention, device_map=device_map)
    if peft_model_path == "":
        model = model
    else:
        model = PeftModel.from_pretrained(model, peft_model_path, device_map=device_map)
        model = model.merge_and_unload()
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 예를 들어, eos_token을 pad_token으로 사용
        # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "right"
    
    # template load
    with open(template_path, "r") as f:
        template_json = json.load(f)
        template = template_json[template_key]
    
    # data load (data_path, template_path, template_key) -> dataloader
    data = dataload_jsonl_reclor(data_path, portion="100%")
    
    model.eval()
    if not ddp and torch.cuda.device_count() > 1:
        # ddp를 쓰지 않고, 
        model.is_parallelizable = True
        model.model_parallel = True
    if ddp and torch.cuda.device_count() > 1:
        sampler = DistributedSampler(data, num_replicas=world_size, rank=local_rank)
        data_loader = DataLoader(data, batch_size=1, sampler=sampler)
    else:
        data_loader = DataLoader(data, batch_size=1, shuffle=False)
        
    inferenced = []
    for ix, x in tqdm(enumerate(data_loader), total=len(data_loader)):  
        # TODO: 데이터 분배 구현하기
        # 이건 전처리 할때 쓰는 방식이고, inference 할 때는 다른 방식을 써야 할 것 같다
        context = x["context"] # str
        question = x["question"]
        answers = x["answers"] # List 
        label = x["label"] # int
        id_string = x["id_string"] # str
        
        # x에 'main_option_correctness' 키가 있는 경우
        if 'main_option_correctness' in x:
            main_option_correctness = x['main_option_correctness'].item()
        else: 
            main_option_correctness = None
            
        # inference
        input_string = template_to_string(template, context, question, answers)
        input_ids = tokenizer(input_string, return_tensors="pt").to('cuda')
        output_ids = model.generate(**input_ids, max_new_tokens=3, num_beams=1, do_sample=False, repetition_penalty=1.0, length_penalty=1.0, no_repeat_ngram_size=4, pad_token_id=tokenizer.pad_token_id)
        output_string = tokenizer.decode(output_ids[0])
        inference = extract_response_from_output(output_string).strip()
        
        answers = [x[0] for x in answers]
        inferenced.append({
            "context": context[0],
            "question": question[0],
            "answers": answers,
            "label": int(label),
            "id_string": id_string[0],
            "input_string": input_string,
            "inference": inference,
            "main_option_correctness": main_option_correctness,
        })
        
        # inform progress (show example)
        if ix % 30 == 0:
            print(f'{ix}-th output:', inference,'\n','label:', label+1)
    
    # save
    if ddp and world_size > 1: # ddp 사용 시 (프로세스마다 결과를 나눠서 저장하고, 마지막에 모아서 저장)

        output_filepath = os.path.join(output_path, f'{output_key}_{local_rank}.jsonl')
        with open(output_filepath, "w") as f:
            for x in inferenced:
                f.write(json.dumps(x) + "\n")
        print(f"Saved partial inference result to {output_filepath}'")

        # Initialize the process group
        dist.init_process_group(backend='nccl')

        # 모든 프로세스가 끝날 때까지 기다림
        torch.distributed.barrier()
        print("barrier passed")
        if local_rank == 0:
            all_inferenced = []
            for rank in range(world_size):
                rank_output_filepath = os.path.join(output_path, f'{output_key}_{rank}.jsonl')
                with open(rank_output_filepath, "r") as f:
                    for line in f:
                        all_inferenced.append(json.loads(line))
            
            # 최종 파일 저장
            final_output_filepath = os.path.join(output_path, f'{output_key}_final.jsonl')
            with open(final_output_filepath, "w") as f:
                for x in all_inferenced:
                    f.write(json.dumps(x) + "\n")
            # 마무리
            
            # 임시 파일 삭제
            for rank in range(world_size):
                rank_output_filepath = os.path.join(output_path, f'{output_key}_{rank}.jsonl')
                os.remove(rank_output_filepath)
                
        torch.cuda.empty_cache()
        dist.destroy_process_group()
    else: # ddp 미사용 시 그냥 저장
        output_filepath = os.path.join(output_path, f'{output_key}.jsonl')
        if output_key == "":
            output_key = data_path.replace(".jsonl", "_inference.jsonl")
        with open(output_filepath, "w") as f:
            for x in inferenced:
                f.write(json.dumps(x) + "\n")
        print(f"Saved inference result to {output_filepath}'")
        return inferenced
    
    
if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(inference)