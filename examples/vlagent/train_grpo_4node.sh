set -x

cd /cpfs/user/zhengziwei/workspace/agent/VeRL-Agent

export HF_HOME=/cpfs/user/zhengziwei/HF_HOME
export PATH=/cpfs/user/zhengziwei/ENV/miniconda3/envs/verl_agent/bin:$PATH
export VLLM_USE_MODELSCOPE=false
export NCCL_DEBUG=WARN
export WANDB_API_KEY=7d84dc21bf59f2e0dd3f214b75a53786cd8fc5d8

PROJECT_NAME=vlagent_vx_new
EXPERIMENT_NAME=multiturn_all21k_0.8acc_-0.2format_0.4tool_4node_ds4
BASE_MODEL=/cpfs/user/zhengziwei/HF_HOME/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/6e6556e8ce728c7b3e438d75ebf04ec93403dc19

VISUAL_DATASET_TRAIN_1=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/train_GQA_1t_fail.parquet
VISUAL_DATASET_TRAIN_2=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/train_llava_focus_1t_fail.parquet
VISUAL_DATASET_TRAIN_3=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/train_spatial_relation_1t_fail.parquet
VISUAL_DATASET_TRAIN_4=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/train_vaw_attribute_1t_fail.parquet

VISUAL_DATASET_VAL_1=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/val_GQA_1t_fail.parquet
VISUAL_DATASET_VAL_2=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/val_llava_focus_1t_fail.parquet
VISUAL_DATASET_VAL_3=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/val_spatial_relation_1t_fail.parquet
VISUAL_DATASET_VAL_4=/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/data/vlagent/parquet/val_vaw_attribute_1t_fail.parquet


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
        data.train_files=[${VISUAL_DATASET_TRAIN_1},${VISUAL_DATASET_TRAIN_2},${VISUAL_DATASET_TRAIN_3},${VISUAL_DATASET_TRAIN_4}] \
        data.val_files=[${VISUAL_DATASET_VAL_1},${VISUAL_DATASET_VAL_2},${VISUAL_DATASET_VAL_3},${VISUAL_DATASET_VAL_4}] \
        data.train_batch_size=256 \
        data.max_prompt_length=4096 \
        data.max_response_length=10240 \
        data.return_raw_chat=True \
        +data.image_downsample=4 \
        algorithm.adv_estimator=grpo \
        algorithm.kl_ctrl.kl_coef=0.0 \
        actor_rollout_ref.model.path=${BASE_MODEL} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.0001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.agent.activate_agent=True \
        actor_rollout_ref.rollout.agent.tool_name_key=env_name \
        actor_rollout_ref.rollout.agent.single_response_max_tokens=4096 \
        actor_rollout_ref.rollout.agent.max_turns=4 \
        actor_rollout_ref.rollout.agent.concurrent_workers=1 \
        actor_rollout_ref.rollout.agent.show_tqdm=True \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard','rl_logging_board','wandb'] \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=8 \
        trainer.test_freq=8 \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.total_epochs=32 \
        +trainer.tensorboard_dir=./logs/tensorboard \
        +trainer.rl_logging_board_dir=./logs/rl_logging_board \
        2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

else
   # Wait until the head node complete
   PORT_RAY=$(sudo netstat -ntlp | grep "/ray")
   while [ -n "${PORT_RAY}" ]; do
      sleep 60s
      PORT_RAY=$(sudo netstat -ntlp | grep "/ray")
   done
   ray stop --force
fi