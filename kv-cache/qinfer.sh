# ## Obtain inference data
task=$1
shots=$4
model=$2
model_arch=$3
base_name=$(basename "${model}")
heavy_ratio=$5
recent_ratio=$6

python -u run_lm_eval_harness.py \
  --input-path results/${task}-${shots}.jsonl \
  --output-path results/${task}-${shots}-${base_name}-qinfer.jsonl \
  --model-name ${model} \
  --model-type ${model_arch} \
  --heavy_ratio ${heavy_ratio} \
  --recent_ratio ${recent_ratio} \
  --qinfer

# --kv_cache ${kv_cache} --recent ${recent}


## Evaluate results
python -u evaluate_task_result.py \
  --result-file results/${task}-${shots}-${base_name}-qinfer.jsonl \
  --task-name ${task} \
  --num-fewshot ${shots} \
  --model-type ${model_arch}

rm results/${task}-${shots}-${base_name}-qinfer.jsonl