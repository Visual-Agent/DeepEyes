set -x

PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="visual_agent_env_v3_model_v0"

export SAVE_CHECKPOINT_DIR=/diancpfs/user/fengyuan/verl_checkpoints
# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

VISUAL_DATASET_TRAIN_1=/cpfs/user/fengyuan/verl_data/visual_agent/visual_agent_train_split1.parquet
VISUAL_DATASET_TRAIN_2=/cpfs/user/fengyuan/verl_data/visual_agent/visual_agent_train_split2.parquet
VISUAL_DATASET_TEST=/cpfs/user/fengyuan/verl_data/visual_agent/visual_agent_test.parquet


REF_MODEL_PATH=/cpfs/user/honglingyi/MODEL/Qwen/Qwen2.5-VL-7B-Instruct
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=[${VISUAL_DATASET_TRAIN_1},${VISUAL_DATASET_TRAIN_2}] \
    data.val_files=${VISUAL_DATASET_TEST} \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=10240 \
    data.return_raw_chat=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0001 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
    actor_rollout_ref.rollout.agent.single_obs_max_length=8192 \
    actor_rollout_ref.rollout.agent.max_turns=7 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=8 \
    trainer.test_freq=8 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.total_epochs=32 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
