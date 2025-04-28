import pandas as pd
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT

# file_path = "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/vl_agent_V1_test_box.parquet"
file_path =["/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/ziwei/parquet/train_GQA_1t_fail.parquet",
            "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/ziwei/parquet/train_llava_focus_1t_fail.parquet",
            "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/ziwei/parquet/train_spatial_relation_1t_fail.parquet",
            "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/ziwei/parquet/train_vaw_attribute_1t_fail.parquet",
            ]
system_prompt = PROMPT.SYSTEM_PROMPT_V2
user_prompt =PROMPT.USER_PROMPT_V2

for path in file_path:
    # 读取 Parquet 文件
    df = pd.read_parquet(path)

    df['env_name'] = 'visual_toolbox_v2'
    df['data_source'] = 'vstar'
    for index, row in df.to_dict('index').items():
        row['prompt'][0]['content'] = system_prompt
        row['prompt'][1]['content'] = '<image>\n' + row['extra_info']['question'] + user_prompt
    # 按列名自定义修改
    # if "prompt" in df.columns:
    #     for item in df:
    #         item['prompt'][0]['content'] = system_prompt
    #         item['prompt'][1]['content'] = '<image>\n' + item['extra_info']['question']

    # 保存修改后的数据
    # output_path = "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/ziwei/parquet/train_vaw_attribute_1t_fail_visual_toolbox_v2.parquet"
    output_path = path.replace(".parquet", "_visual_toolbox_v2.parquet")
    df.to_parquet(output_path, index=False)

    print("自定义修改完成，已保存到:", output_path)

    df = pd.read_parquet(output_path)

    print("修改后可正常读取")
