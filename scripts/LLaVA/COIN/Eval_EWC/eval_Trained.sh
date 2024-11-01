# #!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/ScienceQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/2_eval_textqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/TextVQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/3_eval_ImageNet.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/4_eval_gqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/5_eval_vizwiz.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/6_eval_grounding.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/7_eval_vqav2.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/8_eval_ocrvqa.sh Finetune ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh TextVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/TextVQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh GQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/1_eval_sqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/2_eval_textqa.sh ImageNet ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/ImageNet_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/2_eval_textqa.sh GQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/2_eval_textqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/2_eval_textqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/2_eval_textqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/2_eval_textqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/3_eval_ImageNet.sh GQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/GQA_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/3_eval_ImageNet.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/3_eval_ImageNet.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/3_eval_ImageNet.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/3_eval_ImageNet.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/4_eval_gqa.sh VizWiz ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VizWiz_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/4_eval_gqa.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/4_eval_gqa.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/4_eval_gqa.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/5_eval_vizwiz.sh Grounding ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/Grounding_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/5_eval_vizwiz.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/5_eval_vizwiz.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/6_eval_grounding.sh VQAv2 ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/VQAv2_llava_lora
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/6_eval_grounding.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash ./scripts/LLaVA/COIN/Eval_EWC/7_eval_vqav2.sh OCRVQA ./checkpoints/LLaVA/Instruction/CoIN_EWC_0.5/OCRVQA_llava_lora
