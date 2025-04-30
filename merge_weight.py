from transformers import AutoModelForCausalLM
import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
def average_models(model_path1, model_path2, output_dir):
    """
    平均两个同架构模型的权重
    
    Args:
        model_path1 (str): 第一个模型路径 (HF格式)
        model_path2 (str): 第二个模型路径 (HF格式)
        output_dir (str): 输出目录 (将保存为HF格式)
    """
    # 加载两个模型
    print(f"Loading model from {model_path1}...")
    model1 = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path1, torch_dtype=torch.float16)
    print(f"Loading model from {model_path2}...")
    model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path2, torch_dtype=torch.float16)
    
    # 确保模型架构相同
    # assert model1.config.to_dict() == model2.config.to_dict(), "模型架构不匹配"
    
    # 平均权重
    print("Averaging model weights...")
    with torch.no_grad():
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2, "参数名称不匹配"
            param1.copy_((param1 + param2) / 2)
    
    # 保存模型
    print(f"Saving averaged model to {output_dir}...")
    model1.save_pretrained(output_dir)
    
    # 复制tokenizer文件
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path1)
    tokenizer.save_pretrained(output_dir)
    
    print(f"模型已成功保存到 {output_dir}")

if __name__ == "__main__":
    # 示例用法
    model1_path = "/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/checkpoints/vlagent_think_mh/multiturn_all21k_0.8acc_-0.2format_0.4tool_2node_ds1/global_step_48/actor/huggingface"  # 替换为第一个模型路径
    model2_path = "/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/checkpoints/vlagent_think_mh/multiturn_all21k_0.8acc_-0.2format_0.4tool_4node_ds4/global_step_40/actor/huggingface"  # 替换为第二个模型路径
    output_path = "/cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/checkpoints/vlagent_think_mh/merge_ds1_step48_ds4_step40"  # 输出目录

    os.makedirs(output_path, exist_ok=True)
    average_models(model1_path, model2_path, output_path)