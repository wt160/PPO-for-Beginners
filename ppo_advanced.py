"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

class PPO2:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, envs, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		self.writer = SummaryWriter()
		# launch(tensorboardX, logdir=f"runs")
		print(hyperparameters)
		self.writer.add_text(
			"hyperparameters",
			"|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in hyperparameters.items()])),
		)
		# Make sure the environment is compatible with our code
		# assert(type(envs.single_observation_space) == gym.spaces.Box)
		# assert(type(envs.single_action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.num_envs = 10
		# Extract environment information
		self.envs = envs
		self.obs_dim = envs.single_observation_space.shape[0]
		self.act_dim = envs.single_action_space.shape[0]

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim, 0.01).to(self.device)                                                # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1, 1.0).to(self.device)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(self.device)
		self.cov_mat = torch.diag(self.cov_var).to(self.device)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
			'clipfrac': [],
		}

		self.total_timesteps = 0
		
	# def make_env(self, gym_id, seed, idx, capture_video, run_name):
	# 	def thunk():
	# 		env = gym.make(gym_id)
	# 		env = gym.wrappers.RecordEpisodeStatistics(env)
	# 		if capture_video:
	# 			if idx == 0:                
	# 				env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
	# 		env.seed(seed)
	# 		env.action_space.seed(seed)
	# 		env.observation_space.seed(seed)
	# 		return env
	# 	return thunk

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		self.total_timesteps = total_timesteps
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		update_so_far = 0 # Iterations ran so far
		num_updates = self.total_timesteps // (self.num_envs * self.timesteps_per_batch)


		batch_obs = torch.zeros((self.timesteps_per_batch, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
		batch_acts = torch.zeros((self.timesteps_per_batch, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
		batch_log_probs = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		batch_rews = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		batch_dones = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		batch_V = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		next_obs = self.envs.reset()

		next_obs = torch.Tensor(next_obs).to(self.device)
		next_done = torch.zeros(self.num_envs).to(self.device)
		
		global_step = 0
		while update_so_far < num_updates :                                                                       # ALG STEP 2

			# Keep simulating until we've run more than or equal to specified timesteps per batch
			done = False
			obs = None

			
			# print(next_obs)
			# print(len(next_obs))
			
			
			episode_return = []
			
			for step in range(0, self.timesteps_per_batch):
				global_step += 1 * self.num_envs
				batch_obs[step] = next_obs
				batch_dones[step] = next_done

				# ALGO Logic: action logic
				with torch.no_grad():
					action, logprob = self.get_action(next_obs)
					V = self.get_value(next_obs)
					batch_V[step] = V.flatten()
				batch_acts[step] = action
				batch_log_probs[step] = logprob

				# TRY NOT TO MODIFY: execute the game and log data.
				next_obs, reward, done, info = self.envs.step(action.cpu().numpy()) 

				batch_rews[step] = torch.tensor(reward).to(self.device).view(-1)
				next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)
				
				# get episode length and return
				if 'episode' in info:
					for item in info['episode']:
						if item != None:
							episode_return.append(item["r"])
							# print(f"global_step={global_step}, episodic_return={item['r']}")
							self.writer.add_scalar("charts/episodic_return", item["r"], global_step)
							break


			# batch_rtgs = self.compute_rtgs(batch_rews) 
			nextvalue = self.get_value(next_obs)
			batch_rtgs = self.compute_gae(batch_rews, batch_V, batch_dones, nextvalue, next_done)                                                             # ALG STEP 4

			# Log the episodic returns and episodic lengths in this batch.
			self.logger['episode return'] = episode_return
			# self.logger['batch_lens'] = batch_lens

			bbatch_obs = batch_obs.reshape((-1,) + self.envs.single_observation_space.shape)
			bbatch_log_probs = batch_log_probs.reshape(-1)
			bbatch_acts = batch_acts.reshape((-1,) + self.envs.single_action_space.shape)
			bbatch_rtgs = batch_rtgs.reshape(-1)
			bbatch_V = batch_V.reshape(-1)


			#batch_obs, batch_acts, batch_log_probs, batch_rtgs = self.rollout()                     # ALG STEP 3

			#IMPLEMENTATION DETAIL: learning rate annealing
			trainning_frac = 1.0 - t_so_far / self.total_timesteps
			lrnow = trainning_frac * self.lr
			self.critic_optim.param_groups[0]["lr"] = lrnow
			self.actor_optim.param_groups[0]["lr"] = lrnow

			# Calculate how many timesteps we collected this batch
			t_so_far += self.num_envs * self.timesteps_per_batch

			# Increment the number of iterations
			update_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = update_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(bbatch_obs, bbatch_acts)
			bbatch_V = V.reshape(-1)
			A_k = bbatch_rtgs - bbatch_V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			clipfracs = []
			# This is the loop where we update our network for some n epochs
			for bt in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(bbatch_obs, bbatch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation: 
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - bbatch_log_probs)


				clipfracs += [((ratios - 1.0).abs() > self.clip).float().mean().item()] 
				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
		
				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, bbatch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				# actor_loss.backward()
				
				nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self.logger['clipfrac'] = clipfracs
			self._log_summary()
			
			# Save our model if it's time
			if update_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './' + self.run_name + 'ppo_actor.pth')
				torch.save(self.critic.state_dict(), './' + self.run_name + 'ppo_critic.pth')

	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		# batch_obs = []
		# batch_acts = []
		# batch_log_probs = []
		# batch_rews = []
		# batch_rtgs = []
		# batch_lens = []
		# batch_V = []
		batch_obs = torch.zeros((self.timesteps_per_batch, self.num_envs) + self.envs.single_observation_space.shape).to(self.device)
		batch_acts = torch.zeros((self.timesteps_per_batch, self.num_envs) + self.envs.single_action_space.shape).to(self.device)
		batch_log_probs = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		batch_rews = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		batch_dones = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		batch_V = torch.zeros((self.timesteps_per_batch, self.num_envs)).to(self.device)
		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		done = False
		obs = None

		next_obs = self.envs.reset()
		# print(next_obs)
		# print(len(next_obs))
		next_obs = torch.Tensor(next_obs).to(self.device)
		next_done = torch.zeros(self.num_envs).to(self.device)
		
		episode_return = []
		global_step = 0
		for step in range(0, self.timesteps_per_batch):
			global_step += 1 * self.num_envs
			batch_obs[step] = next_obs
			batch_dones[step] = next_done

			# ALGO Logic: action logic
			with torch.no_grad():
				action, logprob = self.get_action(next_obs)
				V = self.get_value(next_obs)
				batch_V[step] = V.flatten()
			batch_acts[step] = action
			batch_log_probs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
			next_obs, reward, done, info = self.envs.step(action.cpu().numpy()) 

			batch_rews[step] = torch.tensor(reward).to(self.device).view(-1)
			next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)
			
			# get episode length and return
			if 'episode' in info:
				for item in info['episode']:
					if item != None:
						episode_return.append(item["r"])
						# print(f"global_step={global_step}, episodic_return={item['r']}")
						
						break


		# batch_rtgs = self.compute_rtgs(batch_rews) 
		nextvalue = self.get_value(next_obs)
		batch_rtgs = self.compute_gae(batch_rews, batch_V, batch_dones, nextvalue, next_done)                                                             # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['episode return'] = episode_return
		# self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def compute_gae(self, batch_rews, batch_V, batch_dones, next_value, next_done):
		with torch.no_grad():
			advantages = torch.zeros_like(batch_rews).to(self.device)
			batch_rtgs = torch.zeros_like(batch_rews).to(self.device)
			lastgaelam = 0
			t = 0
			batch_rews_size = len([elem for sublist in batch_rews for elem in sublist]) - 1

			#if next step ends the episode, then nextnonterminal = 0, otherwise, advantages[t] depends on advantages[t+1]
			for t in reversed(range(self.timesteps_per_batch)):
				if t == self.timesteps_per_batch - 1:
					nextnonterminal = 1.0 - next_done
					nextvalue = next_value
				else:				
					nextnonterminal = 1.0 - batch_dones[t] 
					nextvalue = batch_V[t+1]
			
				delta = batch_rews[t] + self.gamma * nextvalue * nextnonterminal - batch_V[t]
				advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
				batch_rtgs[t] = advantages[t] + batch_V[t]

		

			return batch_rtgs


		# for t in reversed(range(args.num_steps)):
		# 	if t == args.num_steps - 1:
		# 		nextnonterminal = 1.0 - next_done
		# 		nextvalues = next_value
		# 	else:
		# 		nextnonterminal = 1.0 - dones[t+1]
		# 		nextvalues = values[t+1]
		# 	delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
		# 	advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam 
		# returns = advantages + values

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach(), log_prob.detach()

	def get_value(self, obs):
		V = self.critic(obs).squeeze()
		return V

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
		self.gae_lambda = 0.95
		self.run_name = "hardcore"


		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			print(param)
			print(val)
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed 
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		# avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		# print(self.logger['episode return'])
		avg_ep_rews = np.mean(self.logger['episode return'])
		avg_actor_loss = np.mean([losses.cpu().float().mean() for losses in self.logger['actor_losses']])
		# self.writer.add_scalar("charts/episodic_return", avg_ep_rews, t_so_far)
		# self.writer.add_scalar("charts/episodic_length", item["l"], global_step)
		self.writer.add_scalar("charts/clipfrac", np.mean(self.logger['clipfrac']), t_so_far)
		# self.writer.add_scalar("charts/episodic_return", avg_ep_rews.item(), t_so_far)
		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		# avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)
		
		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['episode return'] = []
		self.logger['actor_losses'] = []
