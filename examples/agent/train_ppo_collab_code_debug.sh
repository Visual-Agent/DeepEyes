set -x

cd /cpfs/user/zhengziwei/workspace/agent/VeRL-Agent

export HF_HOME=/cpfs/user/zhengziwei/HF_HOME
export PATH=/cpfs/user/zhengziwei/ENV/miniconda3/envs/verl_agent/bin:$PATH
export VLLM_USE_MODELSCOPE=false
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATA_DIR=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/collab_code/LeetCodeDataset-v0.3.0

PROJECT_NAME="agent_ppo_collab_code_debug"
EXPERIMENT_NAME=qwen25_0.5b_instruct_debug
BASE_MODEL=/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=16384 \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.single_obs_max_length=512 \
    actor_rollout_ref.rollout.agent.max_turns=10 \
    actor_rollout_ref.rollout.agent.concurrent_workers=4 \
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
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=${BASE_MODEL} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard','rl_logging_board'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    +trainer.tensorboard_dir=./logs/tensorboard \
    +trainer.rl_logging_board_dir=./logs/rl_logging_board \
    2>&1 | tee ./${EXPERIMENT_NAME}.log