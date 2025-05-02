import pandas as pd
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
import io
from PIL import Image
file_path = ["/fs-computility/mabasic/yangminghao/data/ThinkLite-VL-hard-11k/ThinkLite-VL-Hard-11k.parquet"]

system_prompt = PROMPT.SYSTEM_PROMPT_V5
user_prompt =PROMPT.USER_PROMPT_V5

# for path in file_path:
#     # 读取 Parquet 文件
#     df = pd.read_parquet(path)

#     df['env_name'] = 'visual_toolbox_v5'
#     df['data_source'] = 'chart'
#     df['ability'] = 'vl_chart'
#     for index, row in df.to_dict('index').items():
#         row['prompt'][0]['content'] = system_prompt
#         row['prompt'][1]['content'] = row['problem'] + user_prompt
#         row['images'] = [{'bytes':row['image'], "path": ""}]
#         row['reward_model'] = {
#             "style": "model",
#             "ground_truth": row['answer']
#         }
#         row['extra_info'] = {
#             'split': 'train',
#             'index': row['id'],
#             'answer': row['answer'],
#             "question": row['problem'],
#         }

#     # 按列名自定义修改
#     # if "prompt" in df.columns:
#     #     for item in df:
#     #         item['prompt'][0]['content'] = system_prompt
#     #         item['prompt'][1]['content'] = '<image>\n' + item['extra_info']['question']

#     # 保存修改后的数据
#     # output_path = "/fs-computility/mabasic/yangminghao/data/MinghaoYang/mmreasoning/ziwei/parquet/train_vaw_attribute_1t_fail_visual_toolbox_v2.parquet"
#     output_path = path.replace(".parquet", "_visual_toolbox_v5.parquet")
#     df.to_parquet(output_path, index=False)

#     print("自定义修改完成，已保存到:", output_path)

#     df = pd.read_parquet(output_path)

#     print("修改后可正常读取")

for path in file_path:
    # Read only necessary columns from parquet
    df = pd.read_parquet(path, columns=['id', 'problem', 'answer', 'image'])
    
    # Process data in a memory-efficient way
    processed_rows = []
    for _, row in df.iterrows():
        img_bytes = row['image']
        img = Image.open(io.BytesIO(img_bytes))
        width, height = img.size

        # 检查图片尺寸是否过小
        if width < 28 or height < 28:
            print(f"Image too small, skipping: {row['id']}")
            continue  # 跳过当前行

        # 构造处理后的数据
        processed_row = {
            'env_name': 'visual_toolbox_v5',
            'data_source': 'chart',
            'ability': 'vl_chart',
            'prompt': [
                {'content': system_prompt, 'role': 'system'},
                {'content': row['problem'] + user_prompt, 'role': 'user'}
            ],
            'images': [{'bytes': img_bytes, "path": ""}],  # 直接使用原始字节流
            'reward_model': {
                "style": "model",
                "ground_truth": row['answer']
            },
            'extra_info': {
                'split': 'train',
                'index': str(row['id']),
                'answer': row['answer'],
                "question": row['problem'],
            }
        }
        processed_rows.append(processed_row)
    
    # Create new DataFrame with only processed data
    processed_df = pd.DataFrame(processed_rows)
    
    # Save to new path
    output_path = path.replace(".parquet", "_visual_toolbox_v5.parquet")
    processed_df.to_parquet(output_path, index=False)
    
    print(f"Processed and saved to: {output_path}")
    
    # Verification read (optional - remove if not needed)
    verify_df = pd.read_parquet(output_path)
    print("Verification read successful")
    del verify_df  # Free memory
    
    # Explicit cleanup
    del df, processed_df, processed_rows
