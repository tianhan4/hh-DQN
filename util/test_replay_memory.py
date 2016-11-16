import replay_memory
from collections import namedtuple

Config = namedtuple("Config",["memory_size", "history_length", "batch_size", "state_num"])

config = Config(10,2,5,1)
model = replay_memory.ReplayMemory(config)
for i in range(5):
    model.add(i,i*0.1,i*2,i%4==0)
print(model.states,model.actions,model.terminals,model.rewards)
print("sampling")
for i in range(5):
    print(model.sample_one())
for i in range(8):
    model.add(i,i*0.1,i*2,i%4==0)
print(model.states,model.actions,model.terminals,model.rewards)
print("sampling")
for i in range(5):
    print(model.sample_one())

    

