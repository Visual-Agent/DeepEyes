set -x

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export PATH=/cpfs/user/yangminghao/miniforge/envs/agent/bin:$PATH
export WANDB_API_KEY="7297baed6a9f385f68503a4c398d126443c5c747"
wandb login
export DATA_DIR=/cpfs/user/yangminghao/RL/fengyuan/InteractiveRL/data/frozenlake
export OUTPUT_DIR=/cpfs/user/yangminghao/RL/fengyuan/InteractiveRL/ckpt

PROJECT_NAME="agent_ppo_frozenlake"
EXPERIMENT_NAME=qwen25_7b
BASE_MODEL=/cpfs/user/yangminghao/hf_model/Qwen2.5-7B

# BASE_MODEL=/cpfs/user/yangminghao/hf_model/Qwen2.5-0.5B-Instruct
# export VLLM_USE_V1=0

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}

export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=10240 \
    algorithm.adv_estimator=gae \
    algorithm.lam=1 \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=1024 \
    actor_rollout_ref.rollout.agent.single_obs_max_length=256 \
    actor_rollout_ref.rollout.agent.max_turns=10 \
    actor_rollout_ref.rollout.agent.concurrent_workers=4 \
    critic.optim.lr=1e-5 \
    critic.cliprange_value=50 \
    critic.model.path=${BASE_MODEL} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.default_local_dir=${OUTPUT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
