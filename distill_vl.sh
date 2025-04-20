# python api_call_vl.py \
# --input "./data/vlagent/seal_vqa_data/spatial_relation_data.json" \
# --output "./data/vlagent/distilled/spatial_relation_raw.json"

python api_call_vl.py \
--input "./data/vlagent/distilled/spatial_relation_raw.json" \
--output "./data/vlagent/distilled/spatial_relation_1t_fail.json" \
--multi_turn \
--jsonl

python api_call_vl.py \
--input "./data/vlagent/seal_vqa_data/vaw_attribute_data.json" \
--output "./data/vlagent/distilled/vaw_attribute_raw.json"

python api_call_vl.py \
--input "./data/vlagent/distilled/vaw_attribute_raw.json" \
--output "./data/vlagent/distilled/vaw_attribute_1t_fail.json" \
--multi_turn \
--jsonl
