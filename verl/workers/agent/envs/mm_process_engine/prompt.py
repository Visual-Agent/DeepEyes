class PROMPT():
    
    SYSTEM_PROMPT_V1 = """You are a helpful assistant.

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"Path or URL of the image to zoom in."},"bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."}},"required":["image_path","bbox"]}}}
    {"type":"function","function":{"name":"image_rotate_tool","description":"Rotate an image by a specified angle (clockwise or counterclockwise).","parameters":{"type":"object","properties":{"image_path":{"type":"string","description":"Path or URL of the image to be rotated."},"angle":{"type":"integer","description":"Rotation angle in degrees (e.g., 90, 180, 270). Positive values for clockwise, negative for counterclockwise."}},"required":["image_path","angle"]}}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>"""
    # user v1 failed, model do not output toolcall
    USER_PROMPT_V1 = "\nReason in your mind and then give the final answer. Output strictly following the format <think>[your inner thoughts]</think><answer>[your final answer]</answer>."


    # v2: no image_path
#     SYSTEM_PROMPT_V2 = """You are a helpful assistant.

# # Tools

# You may call one or more functions to assist with the user query.

# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."}},"required":["bbox"]}}}
# </tools>

# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>"""

    SYSTEM_PROMPT_V2 = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""

    USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "



    SYSTEM_PROMPT_V3 = ""

    USER_PROMPT_V3 = """\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. 
Otherwise generate a new grouding in JSON format:
```json\n{\n  "function": "zoom_in",\n  "bbox_2d": [x1, y1, x2, y2],\n  "label": "object_name"\n}\n``` 
The zoomed-in image of your grounding will be provided in next turn.
"""

    SYSTEM_PROMPT_V4 = ""

    USER_PROMPT_V4 = """\nIf the current images are insufficient to answer the question, request a zoom-in by providing this tool_call object within tags:
<tool_call>
{"function": "zoom_in", "bbox_2d": [x1, y1, x2, y2], "label": "object_name"}
</tool_call>

The zoomed image will be provided in the next turn. Otherwise, provide your answer within <answer> </answer> tags.
"""



    SYSTEM_PROMPT_V5 = """You are a helpful assistant.

# Tools
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

You may call **one or more** functions to assist with the user query.
**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [8, 40, 50, 150], "label": "the person under the tree"}}  
</tool_call>"""

    # USER_PROMPT_V5 = "\nThink first, call **image_zoom_in_tool** one or more times if needed, i.e., <think>...</think>  <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    # # 看第一轮的rollout，这个会有一些问题，导致模型最后没回答，只是说了一句信息完备，不用调工具了。后续观察score上涨很快，应该自己学会了！
    # TURN_PROMPT_V5 = "\nAbove are the tool responses after calling {}. Think first, continue to call **image_zoom_in_tool** if needed. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed)."
#     TURN_PROMPT_V5_PLUS = """Think in your mind first, <think> Analyze the problem thoroughly. Determine if available information suffices or if tools are needed. Decide whether to call tools one or more times or provide final answer.</think> 
# Then execute one action: <tool_call> tools </tool_call> OR <answer> complete response </answer>
# """
    
    TURN_PROMPT_V5 = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    USER_PROMPT_V5 = TURN_PROMPT_V5