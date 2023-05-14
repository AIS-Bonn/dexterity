from isaacgymenvs.tasks.dexterity.learning.pointcloud_agent import A2CPointcloudAgent
from isaacgymenvs.tasks.dexterity.learning.pointcloud_player import PpoPointcloudPlayerContinuous
import numpy as np
import os
from rl_games.algos_torch import torch_ext
from rl_games.common import common_losses
from rl_games.common.a2c_common import A2CBase, swap_and_flatten01
import time
import torch
import torch.distributed as dist
from typing import *


class DAggerDataset:
    """Dataset collecting observations and corresping teacher actions for DAgger."""
    def __init__(
            self, 
            capacity: int, 
            obs_shape: int, 
            action_shape: int, 
            device: torch.device
        ) -> None:
        self.capacity = capacity
        self.idx = 0
        self.device = device
        self.full = False

        self.obses = torch.empty((capacity, obs_shape)).to(device)
        self.teacher_actions = torch.empty((capacity, action_shape)).to(device)

    def add(self, obs: torch.Tensor, teacher_action: torch.Tensor) -> None:
        """Add new observations and teacher-actions to the dataset.

        Observations and actions are added in a batch-wise fashion. If the dataset is full,
        the oldest observations and actions are overwritten.
        
        Args:
            obs: (torch.Tensor) Observations of the student.
            teacher_action: (torch.Tensor) Actions the teacher has taken in that state.
        """
        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity

        self.full = self.full or remaining_capacity < num_observations

        if remaining_capacity < num_observations:
            self.obses[0:overflow] = obs[-overflow:]
            self.teacher_actions[0:overflow] = teacher_action[-overflow:]

        self.obses[self.idx:self.idx+remaining_capacity] = obs[:remaining_capacity]
        self.teacher_actions[self.idx:self.idx+remaining_capacity] = teacher_action[:remaining_capacity]
        self.idx = (self.idx + remaining_capacity) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = torch.randint(
            0, self.capacity if self.full else self.idx, (batch_size,), 
            device=self.device)
        obses = self.obses[idxs]
        teacher_actions = self.teacher_actions[idxs]
        return obses, teacher_actions


class TAPGAgent(A2CPointcloudAgent):
    """Teacher Augmented Policy Gradient (TAPG) agent."""
    def __init__(self, base_name, params) -> None:
        super().__init__(base_name, params)
        self.ppo_loss_coef = self.config['ppo_loss_coef']
        self.dagger_loss_coef = self.config['dagger_loss_coef']
        self.dagger_batch_size = self.config['dagger_batch_size']

        self.teacher = self._create_teacher(params)
        self.dagger_dataset = DAggerDataset(
            int(self.config['dagger_dataset_capacity']), 
            self.env_info['observation_space'].shape[0], 
            self.env_info['action_space'].shape[0], self.device)

    def _create_teacher(self, params) -> PpoPointcloudPlayerContinuous:
        """Create and restore teacher from checkpoint."""
        # Adjust observation space of the teacher.
        teacher_env_info = {'action_space': self.env_info['action_space'], 
                            'observation_space': self.env_info['teacher_observation_space'],
                            'observation_start_end': self.env_info['teacher_observation_start_end'],
                            'agents': 1}
        
        # Adjust config so that no new vec_env is created.
        teacher_params = params.copy()
        teacher_params['config']['env_info'] = teacher_env_info
        teacher_params['config']['vec_env'] = self.vec_env

        # Student-teacher learning only makes sense when checkpoint for the teacher is specified.
        assert params['teacher_load_path'], "teacher_load_path must be specified to run TAPG."

        # Create and restore teacher.
        teacher = PpoPointcloudPlayerContinuous(teacher_params)
        teacher.restore(params['teacher_load_path'])
        teacher.has_batch_dimension = True
        return teacher

    def calc_gradients(self, input_dict) -> None:
        """Calculates losses and updates agent parameters. Modified to merge ppo_loss and dagger_loss."""
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            ppo_loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            dagger_loss = self.calc_dagger_loss()

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        loss = self.ppo_loss_coef * ppo_loss + self.dagger_loss_coef * dagger_loss

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (ppo_loss, dagger_loss, a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
    def play_steps(self) -> None:
        """Collects agent rollouts. Modified to also store observations and corresponding teacher actions in DAgger dataset."""
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = self.dones.view(self.num_actors, self.num_agents).all(dim=1).nonzero(as_tuple=False)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            # Add observations and corresponding teacher-actions to separate, larger dataset.
            if self.dagger_loss_coef > 0.:
                teacher_actions = self.teacher.get_action(self.obs['teacher_obs'])
                self.dagger_dataset.add(self.obs['obs'], teacher_actions)


        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict
    
    def calc_dagger_loss(self) -> torch.Tensor:
        """Calculates DAgger loss as MSE between student and teacher actions."""
        if self.dagger_loss_coef > 0.:
            obses, teacher_actions = self.dagger_dataset.sample(self.dagger_batch_size)
            student_actions = self._get_differentiable_actions(obses)
            dagger_loss = torch.nn.functional.mse_loss(student_actions, teacher_actions.detach())
        else:
            dagger_loss = torch.tensor(0., device=self.device)
        return dagger_loss

    def _get_differentiable_actions(self, obs_batch):
        """Queries student policy actions in a differentiable manner."""
        # Get actions from the student policy.
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs_batch,
            'rnn_states' : self.states
        }

        if self.model_type == "continuous_a2c_logstd":
            mu, logstd, value, states = self.model.a2c_network(input_dict)
            sigma = torch.exp(logstd)
        elif self.model_type == "continuous_a2c":
            mu, sigma, value, states = self.model.a2c_network(input_dict)
        else:
            assert False
        #distr = torch.distributions.Normal(mu, sigma, validate_args=False)
        #actions_batch = distr.rsample()
        return mu
    
    def train_epoch(self):
        """Runs traning epoch. Modified to log ppo_loss and dagger_loss."""
        A2CBase.train_epoch(self)

        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        ppo_losses = []
        dagger_losses = []
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                ppo_loss, dagger_loss, a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                ppo_losses.append(ppo_loss)
                dagger_losses.append(dagger_loss)
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.rank_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, ppo_losses, dagger_losses, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul

    def train(self) -> Tuple[float, int]:
        """Runs traning. Modified to log ppo_loss and dagger_loss."""
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, ppo_losses, dagger_losses, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                if self.print_stats:
                    step_time = max(step_time, 1e-6)
                    fps_step = curr_frames / step_time
                    fps_step_inference = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                # Log ppo_loss and dagger_loss.
                self.writer.add_scalar('losses/ppo_loss', torch_ext.mean_list(ppo_losses).item(), frame)
                self.writer.add_scalar('losses/dagger_loss', torch_ext.mean_list(dagger_losses).item(), frame)

                # Log ppo and dagger loss weighting terms.
                self.writer.add_scalar('info/last_ppo_loss_coef', self.ppo_loss_coef, frame)
                self.writer.add_scalar('info/last_dagger_loss_coef', self.dagger_loss_coef, frame)

                dagger_replay_size = self.dagger_dataset.capacity if self.dagger_dataset.full else self.dagger_dataset.idx
                self.writer.add_scalar('info/dagger_replay_size', dagger_replay_size, frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num
