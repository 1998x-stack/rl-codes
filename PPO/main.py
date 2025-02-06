# train.py
from ppo import PPO

def main():
    ppo_agent = PPO()
    ppo_agent.train()

if __name__ == "__main__":
    main()