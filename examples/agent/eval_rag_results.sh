set -x

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
DATA_DIR=/cpfs/user/fengyuan/verl_data/r1-searcher

PROJECT_NAME="agent_ppo_debug"
EXPERIMENT_NAME="PPO_new_template_v2"

# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

REF_MODEL_PATH=/cpfs/user/fengyuan/backbone/qwen25/Qwen2.5-7B
REF_MODEL_PATH=/cpfs/user/fengyuan/code/github/verl/checkpoints/agent_ppo_debug/R1-Searcher-32k-v7-retrieval_reward/global_step_160/actor/huggingface
REF_MODEL_PATH=/cpfs/user/fengyuan/code/github/verl/checkpoints/agent_ppo_debug/GRPO-v0-test/global_step_128/actor/huggingface
REF_MODEL_PATH=/cpfs/user/fengyuan/backbone/rag/R1-Searcher
REF_MODEL_PATH=/cpfs/user/fengyuan/backbone/rag/Search-R1
REF_MODEL_PATH=/cpfs/user/fengyuan/code/github/verl/checkpoints/agent_ppo_debug/PPO_new_template_v2/global_step_128/actor/huggingface
REF_MODEL_PATH=/cpfs/user/fengyuan/code/github/verl/checkpoints/agent_ppo_debug/PPO_new_template_v2/global_step_64/actor/huggingface
# REF_MODEL_PATH=/cpfs/user/fengyuan/code/github/verl/checkpoints/agent_ppo_debug/PPO_new_template_gt_return_v2/global_step_88/actor/huggingface

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=10240 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.lam=1.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=2048 \
    actor_rollout_ref.rollout.agent.single_obs_max_length=8192 \
    actor_rollout_ref.rollout.agent.max_turns=9 \
    actor_rollout_ref.rollout.agent.concurrent_workers=4 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    critic.optim.lr=1e-5 \
    critic.cliprange_value=50 \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    trainer.logger=['console'] \
    trainer.val_before_train=True \
    +trainer.val_only=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=32 \
    trainer.test_freq=16 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=32 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
