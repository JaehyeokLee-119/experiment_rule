# 데이터셋 4개의 배열
# 1. RULE_mainq.jsonl
# 2. RULE_subq_all.jsonl
# 3. RULE_500_mainq.jsonl
# 4. RULE_1000_subq.jsonl

# 4개의 데이터셋에 대해 수행하는 코드
file_arr=("RULE_mainq" "RULE_subq_all")

# torchrun 시에는 데이터 parallel
# python inference.py 시에는 데이터 parallel이 아니라 model parallel

# data parallel
for iteration in {0..1}; do
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=4 --master_port=13244 inference.py \
        --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
        --template_path "/hdd/hjl8708/workspace/Inference/templates/mixtral.json" \
        --template_key "prompt_inference_onlynum" \
        --peft_model_path "" \
        --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
        --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
        --output_key "Mixtral_template_${file_arr[iteration]}_Mixtral_4bit" \
        --reduce_memory "4-bit" \
        --flash_attention False 
done 

# model parallel
for iteration in {0..1}; do
    CUDA_VISIBLE_DEVICES=0,1 python inference.py \
        --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
        --template_path "/hdd/hjl8708/workspace/Inference/templates/mixtral.json" \
        --template_key "prompt_inference_onlynum" \
        --peft_model_path "" \
        --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
        --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
        --output_key "Mixtral_template_${file_arr[iteration]}_Mixtral_8bit" \
        --reduce_memory "8-bit" \
        --flash_attention False
done

# # model parallel
# for iteration in {0..3}; do
#     CUDA_VISIBLE_DEVICES=0,1,2 python inference.py \
#         --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
#         --template_path "/hdd/hjl8708/workspace/Inference/templates/mixtral.json" \
#         --template_key "prompt_inference_onlynum" \
#         --peft_model_path "" \
#         --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
#         --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
#         --output_key "Mixtral_template_${file_arr[iteration]}_Mixtral_8bit" \
#         --reduce_memory "8-bit" \
#         --flash_attention False
# done
