class AgentConfig(object):
    scale = 20 # 10000
    display = False

    max_step = 5000 * scale
    memory_size = 100 * scale

    batch_size = 20
    random_start = 30
    cnn_format = 'NCHW'
    discount = 0.95
    target_q_update_learn_step = 2 * scale
    learning_rate = 0.0025
    learning_rate_minimum = 0.0025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    ep_end = 0.1
    ep_start = 1
    ep_end_t = memory_size

    history_length = 1
    train_frequency = 4
    learn_start = 5. * scale

    min_delta = -100
    max_delta = 100

    double_q = True
    dueling = False

    _test_step = 500 * scale
    _save_step = _test_step * 10
    random_seed = 0
    is_train = True
    
    test_ep = None
    n_episode = 10
    n_step = 100

class EnvironmentConfig(object):
    #env_name = 'Breakout-v0'
    env_name = "StochasticMDPEnv"
    screen_width  = 84
    screen_height = 84
    max_reward = 100.
    min_reward = -100.

class DQNConfig(object):
    model_name = 'HQLModel'
    max_stackDepth = 5
    option_num = 3
    option_dim = 20
    goal_pho = 50
    thinking_burden = 5

class M1(AgentConfig, EnvironmentConfig, DQNConfig):
    backend = 'tf'
    env_type = 'detail'
    action_repeat = 1


def get_config(FLAGS):
    config = M1()

    for k, v in FLAGS.__dict__['__flags'].items():
        if k == 'gpu':
            if v == False:
                config.cnn_format = 'NHWC'
            else:
                config.cnn_format = 'NCHW'

        if hasattr(config, k):
            setattr(config, k, v)

    return config