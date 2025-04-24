import re
import random
import requests
import numpy as np
import requests
import base64
import json

from time import sleep
from PIL import Image
from io import BytesIO
from math import ceil, floor

from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents

class VLAgentEnvV3(ToolBase):
    name = "vl_agent_v3"
    
    # user_prompt = "<image>\nHere is the zoomed in image for your grounding region {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer in <answer> </answer> tags. Otherwise generate a new grouding in JSON format, and the zoomed-in image of your grounding will be provided in next turn."
    user_prompt = """\nIf the images provided above are not sufficient to answer the user's question, please generate grouding results in JSON format:
```json
[
    {"bbox_2d": [x1, y1, x2, y2], "label": "label name"}
]
```
The zoomed-in images of your grounding results will be provided in next turn.

Otherwise, please put your final answer in <answer> </answer> tags.
"""
    answer_start = '<answer>'
    answer_end = '</answer>'
    
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

        pattern = re.compile(r'```json\s*([\s\S]*?)```', re.DOTALL)
        action_list = pattern.findall(action_string)
        action_list = [action.strip() for action in action_list]
        if not action_list:
            return '', 0.0, True, {}

        cropped_bbox_list = self.get_bbox_2d(action_list)
        if not cropped_bbox_list:
            user_msg = [{"role": "user", "content": "ZOOM IN ARGUMENTS ARE INVALID"}]
            return user_msg, 0.0, False, {}

        # TODO: modify here and process the final output
        try:
            pil_img = self.multi_modal_data['image'][0]
            cropped_image_list = [pil_img.crop(bbox) for bbox in cropped_bbox_list]
        except Exception as err:
            user_msg = [{"role": "user", "content": "ZOOM IN AREA IS INVALID"}]
            return user_msg, 0.0, False, {}

        if len(cropped_image_list) > 3:
            cropped_image_list = cropped_image_list[:3]
        
        user_msg = '<image>' * len(cropped_image_list) + self.user_prompt
        chat_msg = [{"role": "user", "content": user_msg}]
        obs_dict = {"chat": chat_msg, "multi_modal_data": {"image": cropped_image_list}}
        return obs_dict, 0.0, False, {}


    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'


    def get_bbox_2d(self, action_list):
        if not action_list:
            return None
        
        bbox_list = []
        for action in action_list:
            if not action:
                continue
            try:
                bbox_info = eval(action)
                if isinstance(bbox_info, list):
                    for bbox in bbox_info:
                        bbox_2d = bbox['bbox_2d']
                        self.check_single_bbox(bbox_2d)
                        bbox_list.append(self.maybe_resize_bbox(*bbox_2d))
                else:
                    bbox_2d = bbox_info['bbox_2d']
                    self.check_single_bbox(bbox_2d)
                    bbox_list.append(self.maybe_resize_bbox(*bbox_2d))
            except Exception as err:
                print(f' [ERROR] unexpected {err=}, {action=}')
                continue
        
        return bbox_list


    def check_single_bbox(self, bbox_2d):
        assert isinstance(bbox_2d, list), f"[ERROR] invalid bbox_2d type: {bbox_2d=}"
        assert len(bbox_2d) == 4, f"[ERROR] invalid size for {bbox_2d=}"
        assert self.validate_bbox(*bbox_2d)
    
    
    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 200, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False


    def maybe_resize_bbox(self, left, top, right, bottom):
        height = bottom - top
        width = right - left
        if height < 28 or width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


if __name__ == '__main__':
    tool = VLAgentEnvV3(_name=None, _desc=None, _params=None)
    action_text = """<think> The image shows a building with a steeple and some trees in the foreground. There is a person walking in front of the building, but the details of their clothing are not clear enough to determine the color of their jacket. The image does not provide enough detail to answer the question definitively.\n\nSince the image does not provide sufficient detail to determine the color of the woman\'s jacket, I need to use the zoom_in tool to get a closer look at the person.\n</think>\n<tool_call>\n{"name": "zoom_in", "arguments": {"region": "{\\"bbox_2d\\": [587, 1764, 629, 1860]}"}}\n</tool_call>"""

    observation, reward, done, info = tool.execute(action_string=action_text)
    print (observation)

