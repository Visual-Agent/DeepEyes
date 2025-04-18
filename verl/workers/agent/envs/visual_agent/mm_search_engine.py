import re
import os
import json
import random
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from time import sleep

from playwright.sync_api import sync_playwright, Playwright
from duckduckgo_search import DDGS

from verl.utils.dataset.rl_dataset import process_image
from verl.workers.agent.tool_envs import ToolBase, extract_tool_call_contents

class MMSearchEngine(ToolBase):
    name = "mm_search"

    search_start = "<search>"
    search_end = "</search>"
    browse_start = "<browse>"
    browse_end = "</browse>"
    answer_start = "<answer>"
    answer_end = "</answer>"

    top_k = 5

    def __init__(self, _name, **kwargs):
        self.ddgs = None
        self.chatml_history = []
        super().__init__(name=self.name)

    def execute(self, action_string, **kwargs):
        self.chatml_history.append({
            "role": "assistant",
            "content": action_string,
        })

        answers = extract_tool_call_contents(self.answer_start, self.answer_end, action_string)
        if answers:
            # print(f' [DEBUG] found answer in {action_string=}')
            return '', 0.0, True, {}

        search_list = extract_tool_call_contents(self.search_start, self.search_end, action_string)
        browse_list = extract_tool_call_contents(self.browse_start, self.browse_end, action_string)
        if len(search_list) > 0:
            search_key = ' '.join([item.strip() for item in search_list])
            search_results = self.ddgs.text(search_key, max_results=self.top_k)
            result_text = self.convert_search_to_text(search_results)
            result_text = f"\n<search_result>\n{result_text}\n</search_result>\n"
            return result_text, 0.0, False, {}

        elif len(browse_list) > 0:
            browse_list = [url.strip() for url in browse_list]
            img_list = [self.get_screenshot_from_url(url) for url in browse_list]
            self.multi_modal_data['image'] += img_list

            prompt_list = [f"Screenshot for website {url}\n<image>" for url in browse_list]
            prompt_text = "\n\n".join(prompt_list)
            prompt_text = f"\n<browse_result>\n{prompt_text}\n</browse_result>\n"
            obs = {
                "prompt": prompt_text,
                "multi_modal_data": {"image": img_list},
            }
            print(f' [DEBUG browser] return {len(img_list)} images for {browse_list=}')
            return obs, 0.0, False, {}
        else:
            # print(f' [DEBUG browser] no action_list in {action_string=}')
            return '',  0.0, True, {}

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """
        raw_prompt: list[dict]
        multi_modal_data: List of PIL.Image that has already been preprocessed in rl_dataset
        origin_multi_modal_data: list of PIL.Image that has NOT been preprocessed, original size and format

        For visual agent, all operations are performed on the original image/video
        there is no need to maintain the processed image.
        """
        self.ddgs = DDGS()
        self.chatml_history = raw_prompt.tolist()
        if origin_multi_modal_data is None or not isinstance(origin_multi_modal_data, dict) or 'image' not in origin_multi_modal_data.keys():
            self.multi_modal_data = {"image": []}
        else:
            self.multi_modal_data = origin_multi_modal_data

    def convert_search_to_text(self, search_results):
        search_json_list = []
        for result in search_results:
            docstr = json.dumps(result, ensure_ascii=False, indent=2)
            search_json_list.append(docstr)
        return '\n'.join(search_json_list)

    def get_screenshot_from_url(self, url):
        def run_single(playwright: Playwright):
            chromium = playwright.chromium # or "firefox" or "webkit".
            browser = chromium.launch()
            page = browser.new_page()
            page.goto(url)
            img = page.screenshot()
            browser.close()
            return img

        with sync_playwright() as pw:
            img_bytes = run_single(pw)
            img_pil = Image.open(BytesIO(img_bytes))
        return img_pil