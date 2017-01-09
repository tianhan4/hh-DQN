class AgentConfig(object):
    scale = 100 # 10000

    max_step = 1 * scale
    pre_learn_step = 5000 * scale
    memory_size = 100 * scale

    meta_shut_step = 5000
    batch_size = 32
    random_start = 30
    cnn_format = 'NHWC'
    discount = 0.99
    target_q_update_learn_step = 2 * scale
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale
    ep_end = 0.1
    ep_start = 1
    ep_end_t = scale * 500 #learn_count

    history_length = 1
    s2v_history_length = 1
    train_frequency = 4
    learn_start = 5. * scale


    #w2v part
    s2v_batch_size = 1
    s2v_learning_rate = 0.0025
    s2v_learning_rate_minimum = 0.0001
    s2v_learning_rate_decay = 0.95
    s2v_learning_rate_decay_step = 3000 * scale
    s2v_train_frequency = 6
    state_dim = 3
    window = 6 #TODO: history capacity should be the maximum of history_length and window. for History class.
    neg_sample = 1

    #subgoal part
    shut_step = 5000
    option_num = 15
    #phase1
    subgoal_learning_rate = 0.00025
    subgoal_discount = 0.96
    subgoal_learning_rate_minimum = 0.00025
    subgoal_learning_rate_decay = 0.95
    subgoal_learning_rate_decay_step = 5 * scale
    subgoal_train_frequency = 4
    beta_ep_end = 0.1
    beta_ep_start = 1
    beta_ep_end_t = scale * 1000 #learn_count
    subgoal_ep_end = 0.1
    subgoal_ep_start = 1
    subgoal_ep_end_t = scale * 1000
    #phase2
    subgoal_learning_rate2 = 0.00025
    subgoal_learning_rate_minimum2 = 0.00001
    subgoal_learning_rate_decay2 = 0.96
    subgoal_learning_rate_decay_step2 = 500 * scale
    subgoal_train_frequency2 = 4
    beta_ep_end2 = 0.01
    beta_ep_start2 = 0.1
    beta_ep_end_t2 = scale * 500 #learn_count
    subgoal_ep_end2 = 0.01
    subgoal_ep_start2 = 0.1
    subgoal_ep_end_t2 = scale * 500

    min_delta = -2
    max_delta = 2

    double_q = True
    dueling = False

    _test_step = 500 * scale
    _save_step = _test_step * 10
    random_seed = 0
    is_train = True
    is_pre_model = False
    is_save = True

    test_goal = -1
    test_ep = 0.01
    n_episode = 30
    n_step = 1000

class EnvironmentConfig(object):
    #env_name = 'Breakout-v0'
    env_name = "StochasticMDPEnv3D"
    screen_width  = 84
    screen_height = 84
    max_reward = 100.
    min_reward = -100.

class DQNConfig(object):
    model_name = 'HDQLModel'
    max_stackDepth = 5
    goal_pho = 1
    clip_prob = 0.82


class M1(AgentConfig, EnvironmentConfig, DQNConfig):
    rom_file = "roms/montezuma_revenge.bin"
    backend = 'tf'
    env_type = 'detail'
    action_repeat = 1
    display_screen = False
    mode = "train"
    frame_skip = 4
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

