import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
import TD3
import wandb
from tqdm import tqdm
from time import time

envs_gym = {
	"halfcheetah-random-v2": {'alpha': [1.0]},
	"hopper-random-v2": {'alpha': [1.0]},
	"walker2d-random-v2": {'alpha': [1.0]},
	
	"halfcheetah-medium-v2": {'alpha': [1.0]},
	"hopper-medium-v2": {'alpha': [1.0]},
	"walker2d-medium-v2": {'alpha': [1.0]},
	
	"halfcheetah-medium-replay-v2": {'alpha': [1.0]},
	"hopper-medium-replay-v2": {'alpha': [1.0]},
	"walker2d-medium-replay-v2": {'alpha': [1.0]},

	"halfcheetah-medium-expert-v2": {'alpha': [1.0]},
	"hopper-medium-expert-v2": {'alpha': [1.0]},
	"walker2d-medium-expert-v2": {'alpha': [1.0]},

}


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score


def main(args):
	file_name = f"{args.policy}_{args.env}_{args.alpha}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")
	if not os.path.exists("./results"):
		os.makedirs("./results")
	exist_files = os.listdir('./results/')
	if file_name + '.npy' in exist_files:
		print(f'{file_name} exit, skipping')
		# return
	if not os.path.exists("./infos"):
		os.makedirs("./infos")
	if not os.path.exists("./ebm_models"):
		os.makedirs("./ebm_models")
	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		"alpha": args.alpha,
		"critic_activation": args.critic_activation,
		"layer_norm": args.layer_norm,
	}

	# Initialize policy
	policy = TD3.TD3_BC(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	if args.normalize:
		mean, std = replay_buffer.normalize_states()
	else:
		mean, std = 0, 1

	if not os.path.exists(f"./ebm_models/ebm_{args.env}.pt"):
		for t in tqdm(range(int(args.ebm_steps)), desc='learning ebm'):
			ebm_loss = policy.learn_ebm(replay_buffer, batch_size=args.batch_size)
			if (t + 1) % 5000 == 0:
				e_pos, e_neg = policy.eval_ebm(replay_buffer, batch_size=args.batch_size)
				print(f'ebm_loss: {ebm_loss}, energy_pos: {e_pos}, energy_neg: {e_neg}')
		if args.save_ebm:
			torch.save(policy.ebm, f'./ebm_models/ebm_{args.env}.pt')
	else:
		policy.ebm = torch.load(f'./ebm_models/ebm_{args.env}.pt')
	
	evaluations = []
	info_log = []
	t0 = time()
	for t in (range(int(args.max_timesteps))):
		info = policy.train(replay_buffer, args.batch_size)
		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			info_log.append(info)
			eopch_time = (time() - t0) * 1000 / args.eval_freq
			print(f"Time steps: {t + 1}, epoch_time: {eopch_time:.2f}", info)
			score = eval_policy(policy, args.env, args.seed, mean, std)
			evaluations.append(score)
			wandb.log({'score': score})
			np.save(f"./results/{file_name}", evaluations)
			np.save(f"./infos/{file_name}", info_log)
			if args.save_model: policy.save(f"./models/{file_name}")
			t0 = time()

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="TD3_BC")               # Policy name
	parser.add_argument("--env", default="antmaze-large-play-v0")        # OpenAI gym environment name
	parser.add_argument("--seed", default=42, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=2e6, type=int)   # Max time steps to run environment
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	# TD3
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=5, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--normalize", default=True)
	parser.add_argument("--layer_norm", default=False)
	# EBPC
	parser.add_argument("--alpha", default=1.0, type=float)
	parser.add_argument("--ebm_steps", default=5e5, type=int)   # Max time steps to run environment
	parser.add_argument("--save_ebm", default=True, type=bool)   # Max time steps to run environment

	args = parser.parse_args()

	for seed in [0]:
		args.seed = seed
		for env, env_config in envs_gym.items():
			args.env = env
			for alpha in env_config['alpha']:
				args.alpha = alpha
				for critic_activation in ['tanh']:
					args.critic_activation = critic_activation
					wandb.init(project='EBPC', reinit=True,
							settings=wandb.Settings(code_dir='.', log_internal='./null', _disable_stats=True),
							group='td3+ebm_tanh', mode='offline', save_code=True)
					wandb.config.update(args)
					print(args)
					main(args)