import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json

# 临时修复
# ToolBase.registry = {}

class VisualToolBox(ToolBase):
    name = "visual_toolbox"
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.current_image = None  # To store the current image being processed

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
        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        if not tool_call_match:
            raise ValueError("No tool call found in the action string.")
        
        tool_call_json = tool_call_match.group(1).strip()
        try:
            tool_call = json.loads(tool_call_json)
            return tool_call
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool call: {e}")

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
        try:
            tool_call = self.extract_action(action_string)
            tool_name = tool_call["name"]
            args = tool_call["arguments"]
            
            if tool_name == "image_zoom_in_tool":
                # Zoom in by cropping the image
                image_path = args["image_path"]
                bbox = args["bbox"]
                img = Image.open(image_path)
                cropped_img = img.crop(bbox)
                self.current_image = cropped_img
                
            elif tool_name == "image_rotate_tool":
                # Rotate the image
                image_path = args["image_path"]
                angle = args["angle"]
                img = Image.open(image_path)
                rotated_img = img.rotate(angle)
                self.current_image = rotated_img
                
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            
            # Prepare the observation
            obs = {
                "prompt_token_ids": "<|im_start|>user\nHere is your observation.<|im_end|>\n<|im_start|>assistant\n",
                "multi_model_data": self.current_image,
            }
            reward = 0.1  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            
            return obs, reward, done, info
            
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'DEBUG: Execute WRONG - {str(e)}')
            obs = {
                "prompt_token_ids": f"<|im_start|>user\nError: {str(e)}<|im_end|>\n<|im_start|>assistant\n",
                "multi_model_data": None,
            }
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, *args, **kwargs):
        """
        Reset the environment to its initial state.
        """
        self.current_image = None
        self.reward = 0.0


if __name__ == "__main__":
    # Example usage (for testing)
    tool = VisualToolBox("visual_toolbox", "Tool for image processing", {})
    
    # Test zoom in tool (should return reward=0.1)
    zoom_in_action = """
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"image_path": "test.jpg", "bbox": [10, 10, 100, 100]}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"Zoom in result - Reward: {reward}, Info: {info}")
    
    # Test rotate tool (should return reward=0.1)
    rotate_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(rotate_action)
    print(f"Rotate result - Reward: {reward}, Info: {info}")
    
    # Test invalid JSON (should return reward=0.0)
    invalid_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"image_path": "test.jpg", "angle": 90}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(invalid_action)
    print(f"Invalid JSON result - Reward: {reward}, Info: {info}")
    
    # Test unknown tool (should return reward=0.0)
    unknown_tool_action = """
    <tool_call>
    {"name": "unknown_tool", "arguments": {"param": "value"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(unknown_tool_action)
    print(f"Unknown tool result - Reward: {reward}, Info: {info}")