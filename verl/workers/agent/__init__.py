# NOTE: Env must be imported here in order to trigger metaclass registering
from .envs.rag_engine.rag_engine import RAGEngineEnv
from .envs.rag_engine.rag_engine_v2 import RAGEngineEnvV2
from .envs.visual_agent.vl_agent_v1 import VLAgentEnvV1
from .envs.visual_agent.vl_agent_v2 import VLAgentEnvV2

try:
    from .envs.visual_agent.mm_search_engine import MMSearchEngine
except Exception as err:
    print(f' [ERROR] Failed to register MMSearchEngine : {err=}')
from .envs.frozenlake.frozenlake import FrozenLakeTool
from .envs.collab_code.user_feedback import UserFeedback

from .parallel_env import agent_rollout_loop