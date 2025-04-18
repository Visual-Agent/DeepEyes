import re
import random
import requests
import numpy as np
from PIL import Image
from time import sleep

from verl.utils.dataset.rl_dataset import process_image
from verl.workers.agent.tool_envs import ToolBase

class MMSearchEngine(ToolBase):
    name = "mm_search"
    
    def __init__(self, _name, **kwargs):
        self.chatml_history = []
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        # Format 3: {"prompt": "...", "chat": [{"role": "...", "content": "..."}, ...], "multi_modal_data": ...}
        self.chatml_history.append({"role": "assistant", "content": action_string})

        fake_imgdir = '/cpfs/user/fengyuan/code/github/zero-rl-data/geoguessr/random_streetview_images_pano_v0.0.2/images/image-991-1.png'
        fake_image = Image.open(fake_imgdir)

        user_msg = [{"role": "user", "content": "找不到和您的查询相符的内容或信息。"}]
        self.chatml_history += user_msg
        obs_dict = {"chat": user_msg, "multi_modal_data": {"image": [fake_image]}}
        return user_msg, 0.0, False, {}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """
        raw_prompt: list[dict]
        multi_modal_data: List of PIL.Image that has already been preprocessed in rl_dataset
        origin_multi_modal_data: list of PIL.Image that has NOT been preprocessed, original size and format

        For visual agent, all operations are performed on the original image/video
        there is no need to maintain the processed image.
        """
        self.chatml_history = raw_prompt.tolist()
        if origin_multi_modal_data is None or \
                not isinstance(origin_multi_modal_data, dict) or \
                'image' not in origin_multi_modal_data.keys():
            self.multi_modal_data = {"image": []}
        else:
            self.multi_modal_data = origin_multi_modal_data
