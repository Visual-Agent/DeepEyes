import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
from math import ceil, floor
# 临时修复
# ToolBase.registry = {}

class VisualToolBoxV6(ToolBase):
    name = "visual_toolbox_v6"
    # user_prompt = "Here is the cropped image returned after you calling the function {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise you can continue to call tools within <tool_call></tool_call>."
    user_prompt = PROMPT.USER_PROMPT_V5
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed
        self.origin_multi_modal_data = None


    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None
        
    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            A dictionary with the tool name and arguments.
            
        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.
        
        Args:
            action_string: The string containing the tool call in XML tags.
            
        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """
        
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action_list = self.extract_action(action_string)
        if not action_list:
            return "", 0.0, True, {}
        
        env_response_list, env_image_list = [], []
        
        try:
            for action in action_list:
                tool_call = json.loads(action.strip())
                tool_name = tool_call["name"]
                args = tool_call["arguments"]
        
                if tool_name == "get_focused_image":
                    bbox = args["bbox_2d"]
                    cropped_bbox = self.maybe_resize_bbox(*bbox)
                    if not bbox:
                        raise ValueError(f"GROUNDING ARGUMENTS ARE INVALID")
                    
                    pil_img = self.origin_multi_modal_data['image'][0]
                    ds_width, ds_height = pil_img.width / self.width, pil_img.height / self.height
                    resized_bbox = [int(cropped_bbox[0] * ds_width), int(cropped_bbox[1] * ds_height),
                                    int(cropped_bbox[2] * ds_width), int(cropped_bbox[3] * ds_height)]
                    cropped_image = pil_img.crop(resized_bbox)

                    env_response_list.append("<tool_response>\n" + json.dumps({**{"cropped region": "<image>"}, **args}) + "\n</tool_response>")
                    env_image_list.append(cropped_image)
                
                else:
                    raise ValueError(f"Unknown tool name: {tool_name}")
            
            env_response_list.append(self.user_prompt)
            user_prompt = "<|im_end|>\n<|im_start|>user\n" + "\n".join(env_response_list) + "<|im_end|>\n<|im_start|>assistant\n"
            if env_image_list:
                obs = {"prompt": user_prompt, "multi_modal_data": {"image": env_image_list}}
            else:
                obs = {"prompt": user_prompt}
            reward = 0.0
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG] SUCCESS ACTION {action_string=}')
            return obs, reward, done, info

        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)} {action_string=}')
            obs = "<|im_end|>\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info
        

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = multi_modal_data
        self.origin_multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
        
        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

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
    
if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualToolBoxV5("visual_toolbox", "Tool for image processing", {})
    
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = '<think>\nThe image is a general view of a workspace. There are items on the desk, but there\'s no specific cup or object that can be clearly identified as a material of a cup. If you need a closer look at the material of a cup, we would need a focused image of a cup. However, since the red bounding box option was not provided in the instruction, I cannot call the function to get a focused image of the cup.\n\nThe question is about identifying he material of a cup, but without a specific and prioritized bounding box for the cup, it\'s not possible for me to answer definitively. If there\'s a particular cup you\'re referring to, please provide its bounding box coordinates, and I can provide a focused image of that area.\n</think>  \n<tool_call>\n{"name": "get_focused_image", "arguments": {"bbox_2d": [52, 205, 103, 137], "label": "cup"}}\n</tool_call>'
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - obs: {obs}, Reward: {reward}, Info: {info}")
    