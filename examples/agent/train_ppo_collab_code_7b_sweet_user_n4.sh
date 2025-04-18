set -x

cd /cpfs/user/zhengziwei/workspace/agent/VeRL-Agent

export HF_HOME=/cpfs/user/zhengziwei/HF_HOME
export PATH=/cpfs/user/zhengziwei/ENV/miniconda3/envs/verl_agent/bin:$PATH
export VLLM_USE_MODELSCOPE=false
export NCCL_DEBUG=WARN
# export VLLM_ATTENTION_BACKEND=XFORMERS
export DATA_DIR=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/collab_code/LeetCodeDataset-v0.3.0
export WANDB_API_KEY=7d84dc21bf59f2e0dd3f214b75a53786cd8fc5d8

PROJECT_NAME="agent_ppo_collab_code_spv3"
EXPERIMENT_NAME=qwen_7b_instruct_sweet_user_bs256_no1treward_n4
BASE_MODEL=/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    data.train_files=$DATA_DIR/train_spv3.parquet \
    data.val_files=$DATA_DIR/test_spv3.parquet \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.single_obs_max_length=512 \
    actor_rollout_ref.rollout.agent.max_turns=9 \
    actor_rollout_ref.rollout.agent.concurrent_workers=16 \
    +actor_rollout_ref.rollout.agent.extra_env_info=True \
    actor_rollout_ref.rollout.agent.custom_stop=[] \
    actor_rollout_ref.rollout.agent.user_model.level=sweet \
    actor_rollout_ref.rollout.agent.user_model.url='10.39.0.101' \
    actor_rollout_ref.rollout.agent.user_model.name=qwen-72b-instruct \
    actor_rollout_ref.rollout.agent.user_model.api_key=zzw-114514 \
    actor_rollout_ref.rollout.agent.user_model.temperature=0.7 \
    actor_rollout_ref.rollout.agent.user_model.max_tokens=512 \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
    actor_rollout_ref.rollout.agent.single_obs_max_length=8192 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${BASE_MODEL} \
    critic.model.enable_gradient_checkpointing=True \
    critic.use_dynamic_bsz=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','rl_logging_board','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    +trainer.tensorboard_dir=./logs/tensorboard \
    +trainer.rl_logging_board_dir=./logs/rl_logging_board \
    2>&1 | tee ./${EXPERIMENT_NAME}.log
