export USE_GPU=1
# export MODEL_NAME='/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
# export SERVED_MODEL_NAME='qwen-7b-instruct'
# export MODEL_NAME='/cpfs/user/zhengziwei/HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'
# export SERVED_MODEL_NAME='llama-8b-instruct'
export MODEL_NAME='/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
export SERVED_MODEL_NAME='qwen-32b-instruct'
# export MODEL_NAME='/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31'
# export SERVED_MODEL_NAME='qwen-72b-instruct'
export VLLM_USE_MODELSCOPE=false
export API_KEY="zzw-114514"
export HF_HOME=/cpfs/user/zhengziwei/HF_HOME
export PATH=/cpfs/user/zhengziwei/ENV/miniconda3/envs/vllm/bin:$PATH

echo "$MODEL_NAME"

ifconfig

python -m vllm.entrypoints.openai.api_server \
  --model=$MODEL_NAME \
  --served-model-name=$SERVED_MODEL_NAME \
  --api-key=$API_KEY \
  --trust-remote-code \
  --max-model-len 8192 \
  --tensor-parallel-size=$USE_GPU \
  --port 8000
