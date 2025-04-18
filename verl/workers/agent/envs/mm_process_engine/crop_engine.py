import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from gymnasium.utils import seeding
import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List
from PIL import Image
import re 

# 临时修复
# ToolBase.registry = {}

class CropImageTool(ToolBase):
    name = "crop_image"
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
            description="A tool that crops an image.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "integer", "description": "The action to take (0: none, 1: left, 2: down, 3: right, 4: up)"}
                },
                "required": ["action"]
            }
        )

        
    def extract_action(self, action_string: str) -> str:
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, action_string)
        if matches:
            return matches[-1].strip()
        return ""
    
    def execute(self, action_string, **kwargs):
        """
        Execute the tool functionality by taking an action in the Frozen Lake environment.
        
        Args:
            action: The action to take (0: left, 1: down, 2: right, 3: up)
            
        Returns:
            observation: The current state of the environment (str or rgb_array)
            reward: The reward received after taking the action
            done: Game terminated or not
            info: Additional info
        """
                
        tool = extract_action(action_string)
        cropped_img = img.crop(bbox_2d)
        obs = {
            "prompt_token_ids": "<|im_start|>user\n" + "Here is the cropped image." + "<|im_end|>\n" + "<|im_start|>assistant\n", 
            "multi_model_data": cropped_img,
            }
        done = False
        self.reward = 0.0
        return obs, self.reward, done, {}


    def reset(self, *args, **kwargs):
        """
        Reset the environment to its initial state.
        """
        self.reward = 0



def gen_pil_image(rgb_array):
    pil_image = Image.fromarray(rgb_array)

    pil_image.save('output_image.png')
    pil_image.show()

if __name__ == "__main__":
    use_mm = False
    tool = FrozenLakeTool(_name=None, _desc=None, _params=None, use_mm=use_mm, size=4, is_slippery=False)
    observation, reward, done, info = tool.execute(action="left")
    print(observation, reward, done)
    if not done:
        observation, reward, done, info = tool.execute(action="up")
        print(observation, reward, done)
    if not done:
        observation, reward, done, info = tool.execute(action="up")
        print(observation, reward, done)
    if not done:
        observation, reward, done, info = tool.execute(action="up")
        print(observation, reward, done)
    if not done:
        observation, reward, done, info = tool.execute(action="up")
        print(observation, reward, done)
    if use_mm:
        gen_pil_image(rgb_array=observation)
