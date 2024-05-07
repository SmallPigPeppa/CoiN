RESULT_DIR="./results/CoIN/MiniGPTv2/Grounding"
MODELPATH=$2

deepspeed --include localhost:1,2,3,4,5,6,7 \
    ETrain/Eval/MiniGPT/model_vqa.py \
    --cfg-path ./scripts/MiniGPTv2/Eval/6_Grounding.yaml \
    --image-folder ./cl_dataset \
    --model-path $MODELPATH \
    --answers-file $RESULT_DIR/$1/merge.jsonl \

output_file=$RESULT_DIR/$1/merge.jsonl

python -m ETrain.Eval.LLaVA.CoIN.eval_grounding \
    --test-file ./playground/Instructions_slim/Grounding/test.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$1 \

python -m ETrain.Eval.LLaVA.CoIN.create_prompt \
    --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
    --questions ./playground/Instructions_slim/Grounding/test.json \
    --results $output_file \
    --rule_temp CoIN_Grounding \