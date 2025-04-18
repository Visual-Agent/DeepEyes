set -x

cd /cpfs/user/zhengziwei/workspace/agent/VeRL-Agent

export HF_HOME=/cpfs/user/zhengziwei/HF_HOME
export PATH=/cpfs/user/zhengziwei/ENV/miniconda3/envs/verl_agent/bin:$PATH
export VLLM_USE_MODELSCOPE=false
export NCCL_DEBUG=WARN
# export VLLM_ATTENTION_BACKEND=XFORMERS
export DATA_DIR=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/collab_code/LeetCodeDataset-v0.3.0

PROJECT_NAME="agent_ppo_collab_code"
EXPERIMENT_NAME=qwen_32b_instruct_low_user
BASE_MODEL=/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd

if [ ${RANK} -eq 0 ]; then
   echo "On head node"
   echo ${MASTER_ADDR}
   ray start --head
else
   sleep 30s
   echo "On worker node"
   echo ${MASTER_ADDR}
   ray start --address=${MASTER_ADDR}:6379
fi
if [ "$RANK" = "0" ]; then

    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
        +debug=False \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=10240 \
        data.return_raw_chat=True \
        algorithm.adv_estimator=gae \
        algorithm.lam=1.0 \
        actor_rollout_ref.model.path=${BASE_MODEL} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.agent.activate_agent=True \
        actor_rollout_ref.rollout.agent.single_obs_max_length=512 \
        actor_rollout_ref.rollout.agent.max_turns=10 \
        actor_rollout_ref.rollout.agent.concurrent_workers=16 \
        +actor_rollout_ref.rollout.agent.extra_env_info=True \
        actor_rollout_ref.rollout.agent.custom_stop=[] \
        actor_rollout_ref.rollout.agent.user_model.level=low \
        actor_rollout_ref.rollout.agent.user_model.url='10.39.0.101' \
        actor_rollout_ref.rollout.agent.user_model.name=qwen-72b-instruct \
        actor_rollout_ref.rollout.agent.user_model.api_key=zzw-114514 \
        actor_rollout_ref.rollout.agent.user_model.temperature=0.7 \
        actor_rollout_ref.rollout.agent.user_model.max_tokens=512 \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=${BASE_MODEL} \
        critic.model.enable_gradient_checkpointing=True \
        critic.use_dynamic_bsz=True \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard','rl_logging_board'] \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.total_epochs=15 \
        +trainer.tensorboard_dir=./logs/tensorboard \
        +trainer.rl_logging_board_dir=./logs/rl_logging_board \
        2>&1 | tee ./${EXPERIMENT_NAME}.log

else
   # Wait until the head node complete
   PORT_RAY=$(sudo netstat -ntlp | grep "/ray")
   while [ -n "${PORT_RAY}" ]; do
      sleep 60s
      PORT_RAY=$(sudo netstat -ntlp | grep "/ray")
   done
   ray stop --force
fi