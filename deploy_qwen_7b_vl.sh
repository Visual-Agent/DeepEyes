export MODEL_NAME='/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/checkpoints/vlagent_vx/multiturn_all21k_0.8acc_-0.2format_0.4tool_4node/global_step_32/actor/huggingface'
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