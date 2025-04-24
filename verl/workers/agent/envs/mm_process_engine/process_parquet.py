import pandas as pd
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT

file_path = "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/vl_agent_V1_test_box.parquet"
system_prompt = PROMPT.SYSTEM_PROMPT_V3
user_prompt =PROMPT.USER_PROMPT_V3

# 读取 Parquet 文件
df = pd.read_parquet(file_path)

df['env_name'] = 'visual_toolbox_v3'
df['data_source'] = 'vstar'

# 按列名自定义修改
if "prompt" in df.columns:
    for item in df['prompt']:
        item[0]['content'] = system_prompt
        before, sep, after = item[1]['content'].partition("Question:")
        item[1]['content'] = '<image>\n' + sep + after + user_prompt

# 保存修改后的数据
output_path = "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/vl_agent_V3_test_box.parquet"
df.to_parquet(output_path, index=False)

print("自定义修改完成，已保存到:", output_path)

df = pd.read_parquet(output_path)

print("修改后可正常读取")
