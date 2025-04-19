# curl -X POST http://10.39.17.74:8000/v1/chat/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#         "model": "qwen-vl-7b",
#         "messages": [
#             {
#                 "role": "user",
#                 "content":[
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": "file:///cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg"}
#                     },
#                     {"type": "text", "text": "Is the woman to the left or to the right of the man who is holding the camera?"}
#                 ]
#             }
#         ]
#         }'


img_path="/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg"
img_base64=$(base64 -w 0 "$img_path")

curl -X POST http://10.39.17.74:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vl-7b",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,'"${img_base64}"'"
            }
          },
          {
            "type": "text",
            "text": "Is the woman to the left or to the right of the man who is holding the camera?"
          }
        ]
      }
    ]
  }'