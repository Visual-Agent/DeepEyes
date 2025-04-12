from verl.workers.agent.envs.collab_code.code import code_exec


succ, output = code_exec('import os\nos.system("rm -rf /cpfs/user/zhengziwei/workspace/agent/VeRL-Agent/test_file.md")')

print(succ)
print(output)