Add Support for Huawei Ascend Devices on DeepEyes

# Installation

## Basic Environment Preparation

| software  | version     |
| :-------- | :---------- |
| Python    | ==3.10      |
| CANN      | ==8.1.RC1   |
| torch     | ==2.5.1     |
| torch_npu | ==2.5.1.RC1 |

## Install vllm & vllm-ascend

To ensure proper usage of vllm in verl, it is recommended to install vllm & vllm-ascend via source code compilation.

### Install vllm 0.9.1

```bash
git clone -b v0.9.1 https://github.com/vllm-project/vllm.git 
```

Comment out the dependency on torch in requirements/build.txt

```bash
cd vllm
pip install -r ./requirements/build.txt 
VLLM_TARGET_DEVICE=empty pip install -e .
```

### Install vllm-ascend 0.9.1

```bash
git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git 
cd vllm-ascend

export COMPILE_CUSTOM_KERNELS=1
pip install -e . --no-build-isolation
```

## Install verl

```bash
cd verl
pip install -r requirements-npu.txt 
pip install -e .
```
# Start Training
We use Qwen-2.5-VL-7B-Instruct as our foundation model for RL training. Qwen-2.5-VL-32B-Instruct is also supported.

Step 1: Start a vllm serving of Qwen-2.5-72B-Instruct for llm-as-a-judge verification.
```bash
# download Qwen-2.5-72B-Instruct model
huggingface-cli download --resume-download https://huggingface.co/Qwen/Qwen2.5-72B-Instruct --local-dir /path/to/your/local/filedir --local-dir-use-symlinks False

# start vllm serving
vllm serve /path/to/your/local/filedir \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \ 
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests
```

Step 2: Build a ray cluster for all of the training nodes. Prepare data before starting training. Our training dataset can be downloaded from [huggingface](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k).

Step 3: Use one of the following scripts to start training.

```bash
# your wandb access key here...
wandb login

# the IP and port for your Qwen-2.5-72B-Instruct vllm serving
export LLM_AS_A_JUDGE_BASE="http://your.vllm.machine.ip:18901/v1"

# umber of training nodes
export WORLD_SIZE=8

# config for 7B
bash examples/agent/train_qwen25vl_grpo_agent_npu.sh
```

