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

class VLAgentEnvV2(ToolBase):
    name = "vl_agent_v2"
    
    user_prompt = "This is the zoomed-in image of the region you asked for. The coordinates of the region are {}.\n Please generate the next thought and action. If you can get the answer, please reply with answer in <answer> </answer> tags. Otherwise, you can call the external function again.\n"
    
    action_start = '<tool_call>'
    action_end = '</tool_call>'
    answer_start = '<answer>'
    answer_end = '</answer>'

    # <tool_call>\n{"name": "zoom_in", "arguments": {"object": "woman\'s jacket"}}\n</tool_call>

    chat_template = """<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""
    
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
        cropped_bbox = self.get_bbox_2d(action_list)
        if not cropped_bbox:
            user_msg = self.chat_template.format("ZOOM IN ARGUMENTS ARE INVALID")
            return user_msg, 0.0, False, {}

        # TODO: modify here and process the final output
        try:
            pil_img = self.multi_modal_data['image'][0]
            cropped_image = pil_img.crop(cropped_bbox)
        except Exception as err:
            user_msg = self.chat_template.format("ZOOM IN AREA IS INVALID")
            return user_msg, 0.0, False, {}

        user_msg = "<image>\n" + self.user_prompt.format(cropped_bbox)
        all_user_msg = self.chat_template.format(user_msg)
        obs_dict = {"prompt": all_user_msg, "multi_modal_data": {"image": [cropped_image]}}
        return obs_dict, 0.0, False, {}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width


    def get_bbox_2d(self, action_list):
        if not action_list:
            return None

        for action_string in action_list:
            if not action_string:
                continue
            try:
                action_object = eval(action_string)
                arguments = action_object['arguments']
                if isinstance(arguments, str):
                    arguments = eval(arguments)
                if isinstance(arguments, list):
                    arguments = arguments[0]

                if 'region' in arguments:
                    region = arguments['region']
                else:
                    region = arguments

                if isinstance(region, str):
                    region = eval(region)

                if isinstance(region, dict):
                    bbox = region['bbox_2d']
                elif isinstance(region, list):
                    bbox = region
                else:
                    continue

                if isinstance(bbox, str):
                    bbox = eval(bbox)
                assert isinstance(bbox, list), f"[ERROR] invalid bbox_2d type: {bbox=}"
                assert len(bbox) == 4, f"[ERROR] invalid size for {bbox=}"
                bbox_result = self.maybe_resize_bbox(*bbox)
                if not bbox_result:
                    continue
            except Exception as err:
                print(f' [ERROR vl_agent #1] {err=}')
                continue
        return None
            

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False


    def maybe_resize_bbox(self, left, top, right, bottom):
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)
        if not self.validate_bbox(left, top, right, bottom):
            return None

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
            if not self.validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


if __name__ == '__main__':
    tool = VLAgentEnvV2(_name=None, _desc=None, _params=None)
    action_text = """<think> The image shows a building with a steeple and some trees in the foreground. There is a person walking in front of the building, but the details of their clothing are not clear enough to determine the color of their jacket. The image does not provide enough detail to answer the question definitively.\n\nSince the image does not provide sufficient detail to determine the color of the woman\'s jacket, I need to use the zoom_in tool to get a closer look at the person.\n</think>\n<tool_call>\n{"name": "zoom_in", "arguments": {"region": "{\\"bbox_2d\\": [587, 1764, 629, 1860]}"}}\n</tool_call>"""

    observation, reward, done, info = tool.execute(action_string=action_text)
    print (observation)

