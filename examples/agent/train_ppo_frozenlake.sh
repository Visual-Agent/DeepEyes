set -x

cd /cpfs/user/zhengziwei/workspace/agent/VeRL-Agent

export HF_HOME=/cpfs/user/zhengziwei/HF_HOME
export PATH=/cpfs/user/zhengziwei/ENV/miniconda3/envs/agent/bin:$PATH
# export PATH=/cpfs/user/yangminghao/miniforge/envs/agent/bin:$PATH
export VLLM_USE_MODELSCOPE=false
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATA_DIR=/cpfs/user/yangminghao/RL/fengyuan/InteractiveRL/data/frozenlake
# export OUTPUT_DIR=/cpfs/user/yangminghao/RL/fengyuan/InteractiveRL/ckpt

PROJECT_NAME="agent_ppo_debug_frozenlake"
EXPERIMENT_NAME=qwen25_7b_instruct_debug
BASE_MODEL=/cpfs/user/fengyuan/backbone/qwen25/Qwen2.5-7B-Instruct

# mkdir -p ${OUTPUT_DIR}
# mkdir -p ${OUTPUT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}

# set -x
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.return_raw_chat=True \
    algorithm.adv_estimator=gae \
    algorithm.lam=0.999 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.single_obs_max_length=512 \
    actor_rollout_ref.rollout.agent.max_turns=100 \
    actor_rollout_ref.rollout.agent.concurrent_workers=4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=${BASE_MODEL} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','rl_logging_board'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    +trainer.tensorboard_dir=./logs/tensorboard \
    +trainer.rl_logging_board_dir=./logs/rl_logging_board \
    2>&1 | tee ./${EXPERIMENT_NAME}.log
