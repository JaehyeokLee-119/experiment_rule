# 데이터셋 4개의 배열
# 1. RULE_mainq.jsonl
# 2. RULE_subq_all.jsonl
# 3. RULE_500_mainq.jsonl
# 4. RULE_1000_subq.jsonl

# 4개의 데이터셋에 대해 수행하는 코드
file_arr=("RULE_mainq" "RULE_subq_all") #("ReClor_test" "RULE_mainq" "RULE_subq_all")

# # model parallel: Mixtral-8x7B with target mainQ
# for iteration in {0..0}; do
#     CUDA_VISIBLE_DEVICES=0,1 python inference.py \
#         --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
#         --template_path "/hdd/hjl8708/workspace/Inference/templates/mixtral.json" \
#         --template_key "prompt_inference_with_mainq_onlynum" \
#         --peft_model_path "" \
#         --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
#         --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
#         --output_key "Mixtral_template_including_targetq_${file_arr[iteration]}_Mixtral_8bit" \
#         --reduce_memory "8-bit" \
#         --flash_attention False \
#         --including_target_q True
# done

# model parallel
for iteration in {0..1}; do
    CUDA_VISIBLE_DEVICES=0,1,2 python inference.py \
        --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
        --template_path "/hdd/hjl8708/workspace/Inference/templates/mixtral.json" \
        --template_key "prompt_inference_cot1" \
        --peft_model_path "" \
        --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
        --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
        --output_key "Mixtral_template_cot1_(ThreeParagraphReasoning)_${file_arr[iteration]}_Mixtral_half" \
        --reduce_memory "half-precision" \
        --flash_attention False \
        --including_target_q False \
        --max_new_tokens 500
done

# model parallel
for iteration in {0..1}; do
    CUDA_VISIBLE_DEVICES=0,1,2 python inference.py \
        --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
        --template_path "/hdd/hjl8708/workspace/Inference/templates/mixtral.json" \
        --template_key "prompt_inference_cot2" \
        --peft_model_path "" \
        --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
        --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
        --output_key "Mixtral_template_cot2_(OnlyLetsThinkStepbyStep)_${file_arr[iteration]}_Mixtral_half" \
        --reduce_memory "half-precision" \
        --flash_attention False \
        --including_target_q False \
        --max_new_tokens 500
done
# 
# torchrun 시에는 데이터 parallel#
# python inference.py 시에는 데이터 parallel이 아니라 model parallel

# # data parallel
# for iteration in {0..1}; do
#     CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=13244 inference.py \
#         --model_name "mistralai/Mixtral-8x7B-Instruct-v0.1" \
#         --template_path "/hdd/hjl8708/workspace/Inference/templates/mixtral.json" \
#         --template_key "prompt_inference_onlynum" \
#         --peft_model_path "" \
#         --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
#         --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
#         --output_key "Mixtral_template_${file_arr[iteration]}_Mixtral_4bit" \
#         --reduce_memory "4-bit" \
#         --flash_attention False
# done 
#" \

# for iteration in {0..1}; do # Llama-2-70B 4bit (DDP)
#     CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=13244 inference.py \
#         --model_name "meta-llama/Llama-2-70b-chat-hf" \
#         --template_path "/hdd/hjl8708/workspace/Inference/templates/alpaca.json" \
#         --template_key "prompt_inference_onlynum" \
#         --peft_model_path "" \
#         --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
#         --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
#         --output_key "Alpaca_template_${file_arr[iteration]}_Llama-70B-4bit" \
#         --reduce_memory "4-bit" \
#         --flash_attention False
# done

# for iteration in {0..1}; do # Llama-2-70B 8bit
#     CUDA_VISIBLE_DEVICES=0,1,2 python inference.py \
#         --model_name "meta-llama/Llama-2-70b-chat-hf" \
#         --template_path "/hdd/hjl8708/workspace/Inference/templates/alpaca.json" \
#         --template_key "prompt_inference_onlynum" \
#         --peft_model_path "" \
#         --data_path "/hdd/hjl8708/workspace/Data/RULE/${file_arr[iteration]}.jsonl" \
#         --output_path "/hdd/hjl8708/workspace/Inference/Result/" \
#         --output_key "Alpaca_template_${file_arr[iteration]}_Llama-70B-8bit" \
#         --reduce_memory "8-bit" \
#         --flash_attention False
# done

# # ! Mixtral 8x7B
# for iteration in {0..1}; do
#     CUDA_VISIBLE_DEVICES=0,1 python inference.py \
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