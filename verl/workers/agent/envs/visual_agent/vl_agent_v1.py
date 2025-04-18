import re
import random
import requests
import numpy as np
from time import sleep
import requests
from PIL import Image
from io import BytesIO
import base64
import json

from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents

class VLAgentEnvV1(ToolBase):
    name = "vl_agent"
    
    # url_query = "http://0.0.0.0:24000/query"
    query_urls = [
        "http://10.39.0.82:24000/query",
        # "http://10.39.0.82:24100/query",
        "http://10.39.0.83:24000/query",
        "http://10.39.0.84:24000/query",
        "http://10.39.0.81:24000/query",
    ]

    user_prompt = "This is the zoomed-in image of the object you asked for.\n Please generate the next thought and action. If you can get the answer, please reply with answer in <answer> </answer> tags. Otherwise, you can call the external function again.\n"

    action_start = '<tool_call>'
    action_end = '</tool_call>'
    answer_start = '<answer>'
    answer_end = '</answer>'

    chat_template = """<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""

    # <tool_call>\n{"name": "zoom_in", "arguments": {"object": "woman\'s jacket"}}\n</tool_call>

    def __init__(self, _name, _desc, _params, **kwargs):
        self.chatml_history = []
        self.multi_modal_data = None
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
        if answers:
            # print(f' [DEBUG] found answer in {action_string=}')
            return '', 0.0, True, {}

        action_list = extract_tool_call_contents(self.action_start, self.action_end, action_string)
        if not action_list:
            # print(f' [DEBUG] no action_list in {action_string=}')
            return '',  0.0, True, {}

        action_list = [action.strip() for action in action_list]
        pil_img = self.multi_modal_data['image'][0]
        pil_img_str = self.encode_image(pil_img)

        query_results = self.batch_query(action_list, pil_img_str)
        print(f' [DEBUG vl_agent] {query_results=}')

        if 'output' not in query_results or not query_results['output']:
            print(f' [WARNING] {action_list=} has no query result : {action_list}')
            user_msg = self.chat_template.format("ZOOM IN RESULT IS EMPTY")
            return user_msg, 0.0, False, {}

        if isinstance(query_results['output'], list):
            query_results = query_results['output'][0]
        elif isinstance(query_results['output'], dict):
            query_results = query_results['output']
        else:
            print(f' [WARNING] invalid type for {query_results=}')
            user_msg = self.chat_template.format("ZOOM IN RESULT IS INVALID")
            return user_msg, 0.0, False, {}

        if not self.validate_bbox(query_results):
            user_msg = self.chat_template.format("ZOOM IN RESULT IS INVALID")
            return user_msg, 0.0, False, query_results

        cropped_pil_image = self.crop_img(pil_img, query_results)
        user_msg = "<image>\n" + self.user_prompt
        all_user_msg = self.chat_template.format(user_msg)
        obs_dict = {"prompt": all_user_msg, "multi_modal_data": {"image": [cropped_pil_image]}}
        return obs_dict, 0.0, False, {}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'


    def encode_image(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = buffered.getvalue()
        encoded_image = base64.b64encode(img_str)
        encoded_image_text = encoded_image.decode("utf-8")
        return encoded_image_text


    def validate_bbox(self, query_results):
        try:
            bbox = query_results['bbox_2d']
            left, top, right, bottom = bbox
            assert left < right and bottom > top, f'invalid shape for {bbox=}'
            return True
        except Exception as err:
            print(f' [ERROR] {err}')
            return False


    def crop_img(self, pil_img, query_results):
        bbox = query_results['bbox_2d']
        left, top, right, bottom = bbox
        cropped_image = pil_img.crop((left, top, right, bottom))
        return cropped_image


    def batch_query(self, queries, pil_img_str, max_retry=32):
        try:
            query = queries[0]
            query = eval(query)
            obj_name = query['arguments']['object']
        except Exception as err:
            print(f' [WARNING] invalid model predicted query: {queries}')
            return {}

        payload = {"image": pil_img_str, "text": obj_name}
        for it in range(max_retry):
            try:
                response = requests.post(random.choice(self.query_urls), json=payload)
                if response.status_code == 200:
                    resjson = response.json()
                    return resjson
                else:
                    continue
            except Exception as err:
                print(f' [ERROR] err={err} -- retry for {it}')
                sleep(random.uniform(0.1, 1.0))
                continue
        return {}


if __name__ == '__main__':
    tool = VLAgentEnvV1(_name=None, _desc=None, _params=None)
    action_text = '<think> The image shows a building with a steeple and some trees in the foreground. There is a person walking in front of the building, but the details of their clothing are not clear enough to determine the color of their jacket. The resolution and angle of the photo do not provide sufficient detail to identify the color accurately.\n\nSince the image does not provide enough detail to determine the color of the woman\'s jacket, I need to use the zoom_in tool to get a closer look at her jacket.\n</think>\n<tool_call>\n{"name": "zoom_in", "arguments": {"object": "woman\'s jacket"}}\n</tool_call>'

    observation, reward, done, info = tool.execute(action_string=action_text)
    print (observation)

