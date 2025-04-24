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
    SYSTEM_PROMPT_V2 = """You are a helpful assistant.

    # Tools

    You may call one or more functions to assist with the user query.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox).","parameters":{"type":"object","bbox":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."}},"required":["bbox"]}}}
    {"type":"function","function":{"name":"image_rotate_tool","description":"Rotate an image by a specified angle (clockwise or counterclockwise).","parameters":{"type":"object", "angle":{"type":"integer","description":"Rotation angle in degrees (e.g., 90, 180, 270). Positive values for clockwise, negative for counterclockwise."}},"required":["angle"]}}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {"name": <function-name>, "arguments": <args-json-object>}
    </tool_call>"""

    USER_PROMPT_V2 = "\nThink first, call tools if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (optional)  <answer>...</answer> "



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
