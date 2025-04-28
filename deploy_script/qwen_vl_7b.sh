export MODEL_NAME='/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/6e6556e8ce728c7b3e438d75ebf04ec93403dc19'
export SERVED_MODEL_NAME='qwen-vl-7b'
export VLLM_USE_MODELSCOPE=false
export API_KEY="zzw-114514"
export HF_HOME=/cpfs/user/zhengziwei/HF_HOME
export PATH=/cpfs/user/zhengziwei/ENV/miniconda3/envs/verl_agent/bin:$PATH

echo "$MODEL_NAME"

ifconfig

python -m vllm.entrypoints.openai.api_server \
  --model=$MODEL_NAME \
  --served-model-name=$SERVED_MODEL_NAME \
  --api-key=$API_KEY \
  --trust-remote-code \
  --max-model-len 32768 \
  --tensor-parallel-size=1 \
  --allowed-local-media-path /cpfs \
  --limit-mm-per-prompt image=10 \
  --port 8000