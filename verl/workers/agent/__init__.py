# NOTE: Env must be imported here in order to trigger metaclass registering
from .envs.rag_engine.rag_engine import RAGEngineEnv
from .envs.rag_engine.rag_engine_v2 import RAGEngineEnvV2
from .envs.visual_agent.vl_agent_v1 import VLAgentEnvV1
from .envs.visual_agent.vl_agent_v2 import VLAgentEnvV2
from .envs.mm_process_engine.visual_toolbox import VisualToolBox

try:
    from .envs.visual_agent.mm_search_engine import MMSearchEngine
except Exception as err:
    print(f' [ERROR] Failed to register MMSearchEngine : {err=}')

try:
    from .envs.frozenlake.frozenlake import FrozenLakeTool
except Exception as err:
    print(f' [ERROR] Failed to register FrozenLakeTool : {err=}')

from .parallel_env import agent_rollout_loop