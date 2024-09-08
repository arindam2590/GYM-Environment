from dqn_agent_torch import DQNAgent


def main():
    env_name = "FrozenLake-v1"
    episodes = 25000
    render = False
    is_training = False

    agent = DQNAgent(env_name, is_training, render)
    agent.play_game(episodes)


if __name__ == '__main__':
    main()
