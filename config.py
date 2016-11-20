class AgentConfig(object):
    scale = 20 # 10000
    display = False

    max_step = 5000 * scale
    memory_size = 100 * scale

    batch_size = 32
    random_start = 30
    cnn_format = 'NHWC'
    discount = 0.99
    target_q_update_learn_step = 8
    learning_rate = 0.025
    learning_rate_minimum = 0.025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    ep_end = 0.1
    ep_start = 1
    ep_end_t = memory_size *5

    history_length = 1
    train_frequency = 2
    learn_start = 5. * scale

    min_delta = -100
    max_delta = 100

    double_q = True
    dueling = False

    _test_step = 50 * scale
    _save_step = _test_step * 10
    random_seed = 0
    is_train = True
    
    test_ep = None
    n_episode = 10
    n_step = 100

class EnvironmentConfig(object):
    #env_name = 'Breakout-v0'
    env_name = "ALEEnvironment"
    screen_width  = 84
    screen_height = 84
    max_reward = 100.
    min_reward = -100.

class DQNConfig(object):
    model_name = 'DHQLModel'
    max_stackDepth = 10
    option_num = 20
    goal_pho = 10
    shut_step = 300


class M1(AgentConfig, EnvironmentConfig, DQNConfig):
    rom_file = "roms/breakout.bin"
    mode = "train"
    backend = 'tf'
    env_type = 'detail'
    action_repeat = 1
    display_screen = 0
    frame_skip = 2
    color_averaging = True
    record_screen_path = None
    record_sound_filename = None
    minimal_action_set = True

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
