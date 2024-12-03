# #!/bin/bash

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/ScienceQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/2_eval_textqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/TextVQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/3_eval_ImageNet.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/4_eval_gqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/5_eval_vizwiz.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/6_eval_grounding.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/7_eval_vqav2.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/8_eval_ocrvqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora

# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/TextVQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN_LWF/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh GQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_LWF/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_LWF/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_LWF/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/1_eval_sqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/2_eval_textqa.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN_LWF/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/2_eval_textqa.sh GQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/2_eval_textqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_LWF/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/2_eval_textqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_LWF/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/2_eval_textqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_LWF/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/2_eval_textqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/3_eval_ImageNet.sh GQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/3_eval_ImageNet.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_LWF/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/3_eval_ImageNet.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_LWF/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/3_eval_ImageNet.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_LWF/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/3_eval_ImageNet.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/4_eval_gqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_LWF/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/4_eval_gqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_LWF/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/4_eval_gqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_LWF/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/4_eval_gqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/5_eval_vizwiz.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_LWF/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/5_eval_vizwiz.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_LWF/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/5_eval_vizwiz.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/6_eval_grounding.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_LWF/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/6_eval_grounding.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_LWF/7_eval_vqav2.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_LWF/OCRVQA_llava_lora
