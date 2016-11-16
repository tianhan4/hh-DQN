import envs.mdp as env

keymap = ["w","a","s","d"]
game = env.StochasticMDPEnv()
game.reset()
i = 0
while True:
    action = keymap.index(input())
    reward, state, terminal = game.step(action)
    score = game.score
    print(i,reward,state,terminal,score)
    print("step:%d reward:%f, state:%d terminal:%s score: %f"%(i,reward,state,terminal,score))
    if terminal:
        game.reset()




