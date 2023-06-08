import numpy as np
import os
from rl_games.algos_torch import torch_ext
from rl_games.common.a2c_common import swap_and_flatten01
import time
import torch
import torch.distributed as dist
from torch import optim
from typing import *
import random
from collections import defaultdict
from rl_games.algos_torch import central_value
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import a2c_common, datasets
from torch import optim

from rl_games.common.player import BasePlayer
from rl_games.algos_torch.players import PpoPlayerContinuous



class TeacherDataset:
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
        self.teacher_values = torch.empty((capacity, 1)).to(device)

    def add(self, obs: torch.Tensor, teacher_action: torch.Tensor, teacher_values: torch.Tensor) -> None:
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
        self.teacher_values[self.idx:self.idx+remaining_capacity] = teacher_values[:remaining_capacity]
        self.idx = (self.idx + remaining_capacity) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = torch.randint(
            0, self.capacity if self.full else self.idx, (batch_size,), 
            device=self.device)
        obses = self.obses[idxs]
        teacher_actions = self.teacher_actions[idxs]
        teacher_values = self.teacher_values[idxs]
        return obses, teacher_actions, teacher_values
    

class AIDAgent(A2CAgent):
    def __init__(self, base_name, params) -> None:
        self.st_cfg = params['student_teacher']
        self._acquire_student(base_name, params)
        self.teacher = self._acquire_teacher(params)

        self.train_result = (torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), self.last_lr, torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]))
        self.imitate_results = defaultdict(list)

        if self.st_cfg['method'] == 'aid':
            self.imitation_optimizer = optim.Adam(self.model.a2c_network.imitation_head.parameters(), float(self.st_cfg['imitation_lr']))
        elif self.st_cfg['method'] in ['dagger', 'cri']:
            self.imitation_optimizer = optim.Adam(self.model.parameters(), float(self.st_cfg['imitation_lr']))

        else:
            pass
        
    def _acquire_student(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'observation_start_end': self.env_info['observation_start_end'],
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'agent_params': params,
        }

        self.model_type = params['model']['name']

        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        self.algo_observer.after_init(self)
        
    def _acquire_teacher(self, params) -> PpoPlayerContinuous:
        """Create and restore teacher from checkpoint."""
        self.teacher_cfg = params['teacher']
        
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
        assert params['teacher_load_path'], "teacher_load_path must be specified to run AID."

        # Create and restore teacher.
        teacher = PpoPlayerContinuous(teacher_params)
        teacher.restore(params['teacher_load_path'])
        teacher.has_batch_dimension = True

        # Create dataset for teacher.
        self.teacher_dataset = TeacherDataset(
            int(self.teacher_cfg['dataset_capacity']),
            self.env_info['observation_space'].shape[0],
            self.env_info['action_space'].shape[0], self.device)
        return teacher

    @property
    def teacher_action_prob(self) -> float:
        return self.teacher_cfg['action_prob_init'] * self.teacher_cfg['action_prob_decay'] ** self.epoch_num

    def play_steps(self) -> None:
        """Collects agent rollouts. Modified to also store observations and corresponding teacher actions in TeacherDataset."""
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            teacher_actions = self.teacher.get_action(self.obs['teacher_obs'], is_determenistic=True)
            teacher_input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : self.obs['teacher_obs'],
            'rnn_states' : self.states
            }
            teacher_res_dict = self.teacher.model(teacher_input_dict)
            teacher_values = teacher_res_dict['values']
            self.teacher_dataset.add(self.obs['obs'], teacher_actions, teacher_values)

            # Log distance between student and teacher actions.
            self.imitate_results['info/student_teacher_action_l2_distance'].append(torch.nn.functional.mse_loss(res_dict['mus'], teacher_actions).item())
            self.imitate_results['info/student_teacher_action_l1_distance'].append(torch.nn.functional.l1_loss(res_dict['mus'], teacher_actions).item())

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            # If sampled float is smaller than teacher_action_prob, use teacher actions instead of student actions.
            if random.random() < self.teacher_action_prob:
                res_dict['actions'] = teacher_actions

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

    def _get_differentiable_actions(self, obs_batch, is_deterministic=True, return_values=False):
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
        if is_deterministic:
            if return_values:
                return mu, value
            return mu
        else:
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if return_values:
                return distr.rsample(), value
            return distr.rsample()
    
    def calc_gradients(self, input_dict) -> None:
        """Calculates losses and updates agent parameters."""

        if self.st_cfg['method'] == 'none':
            return super().calc_gradients(input_dict)
        
        else:
            return getattr(self, 'calc_' + self.st_cfg['method'] + '_gradients')(input_dict)
        
    def calc_dagger_gradients(self, input_dict) -> None:
        dagger_losses = []
        for mini_ep in range(self.st_cfg['imitation_miniepochs']):
            dagger_loss = self.calc_dagger_loss()
            self.imitation_optimizer.zero_grad()
            dagger_loss.backward()
            self.imitation_optimizer.step()
            dagger_losses.append(dagger_loss.item())
        self.imitate_results['losses/dagger_l2'].append(sum(dagger_losses) / len(dagger_losses))

    def calc_dagger_loss(self) -> torch.Tensor:
        """Calculates DAgger loss as MSE between student and teacher actions."""
        obses, teacher_actions, teacher_values = self.teacher_dataset.sample(self.teacher_cfg['batch_size'])
        student_actions = self._get_differentiable_actions(obses)
        dagger_loss = torch.nn.functional.mse_loss(student_actions, teacher_actions.detach())
        return dagger_loss

    def calc_aid_gradients(self, input_dict) -> None:
        # Calculated imitation head gradients.
        self.calc_imitation_head_gradients()
        # Calculate regular policy gradients.
        super().calc_gradients(input_dict)

    def calc_cri_gradients(self, input_dict) -> None:
        # Calculate regular policy gradients.
        super().calc_gradients(input_dict)

        imitation_losses = []
        mean_teacher_values = []
        mean_student_values = []
        for _ in range(5):
            obses, teacher_actions, teacher_values = self.teacher_dataset.sample(self.teacher_cfg['batch_size'])

            student_actions, student_values = self._get_differentiable_actions(obses, return_values=True)

            teacher_knows_better = (teacher_values.detach() > student_values.detach()).float().repeat(1, 11)

            #print("teacher_knows_better:", teacher_knows_better)
            #print("teacher_knows_better.shape:", teacher_knows_better.shape)

            action_diff = teacher_knows_better * (teacher_actions - student_actions)

            imitation_loss = torch.norm(action_diff, p=2, dim=1).mean()

            #print("imitation_loss:", imitation_loss)

            self.imitation_optimizer.zero_grad()
            imitation_loss.backward()
            self.imitation_optimizer.step()

            imitation_losses.append(imitation_loss.item())
            mean_student_values.append(student_values.mean().item())
            mean_teacher_values.append(teacher_values.mean().item())

        self.imitate_results['losses/imitation_l2'].append(sum(imitation_losses) / len(imitation_losses))
        self.imitate_results['values/student'].append(sum(mean_student_values) / len(mean_student_values))
        self.imitate_results['values/teacher'].append(sum(mean_teacher_values) / len(mean_teacher_values))
        self.imitate_results['values/frac_teacher_knows_better'].append(teacher_knows_better[0].mean().item() / teacher_knows_better.shape[0])

    def calc_imitation_head_gradients(self) -> None:
        imitation_losses = []
        for mini_ep in range(self.st_cfg['imitation_miniepochs']):
            imitation_loss = self.calc_imitation_loss()
            self.imitation_optimizer.zero_grad()
            imitation_loss.backward()
            self.imitation_optimizer.step()
            imitation_losses.append(imitation_loss.item())
        self.imitate_results['losses/imitation_l2'].append(sum(imitation_losses) / len(imitation_losses))

    def calc_imitation_loss(self) -> torch.Tensor:
        obses, teacher_actions, teacher_values = self.teacher_dataset.sample(self.teacher_cfg['batch_size'])
        
        # Normalize observations because I will also get normalized student observations when queried in the policies forward call.
        obses = self.model.norm_obs(obses)

        imitation_head_actions = self.model.a2c_network.imitation_head(obses)
        imitation_loss = torch.nn.functional.mse_loss(imitation_head_actions, teacher_actions.detach())
        return imitation_loss

    def train(self) -> Tuple[float, int]:
        """Runs traning."""
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
            self.epoch_num = epoch_num
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
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

                for k, v in self.imitate_results.items():
                    self.writer.add_scalar(k, sum(v) / len(v), frame)
                self.imitate_results = defaultdict(list)  # Reset imitate_results
                self.writer.add_scalar('info/last_teacher_action_prob', self.teacher_action_prob, frame)
                teacher_replay_size = self.teacher_dataset.capacity if self.teacher_dataset.full else self.teacher_dataset.idx
                self.writer.add_scalar('info/teacher_replay_size', teacher_replay_size, frame)
                
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


    def train_epoch(self):
        from rl_games.common.a2c_common import A2CBase
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

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
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

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul



class AIDPlayerContinuous(PpoPlayerContinuous):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            #'observation_start_end': self.env_info['observation_start_end'],
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
            'agent_params': params,
        }

        self.model_type = params['model']['name']
        self.model = self.network.build(build_config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

# class FeatureMimicking:
#     def calc_feature_mimicking_gradients(self) -> None:
#         self.calc_imitation_head_gradients()

#     def calc_imitation_head_gradients(self) -> None:
#         imitation_losses = []
#         for mini_ep in range(self.kd_cfg['imitation_miniepochs']):
#             imitation_loss = self.calc_imitation_loss()
#             self.imitation_optimizer.zero_grad()
#             imitation_loss.backward()
#             self.imitation_optimizer.step()
#             imitation_losses.append(imitation_loss.item())
#         self.kd_results['losses/imitation_l2'].append(sum(imitation_losses) / len(imitation_losses))

#     def calc_imitation_loss(self) -> torch.Tensor:
#         obses, teacher_actions = self.teacher_dataset.sample(self.teacher_cfg['batch_size'])
#         imitation_head_actions = self.imitation_head(obses)
#         imitation_loss = torch.nn.functional.mse_loss(imitation_head_actions, teacher_actions.detach())
#         return imitation_loss
    

#     def get_action_values(self, obs):
#         if self.kd_cfg['distillation_mode'] == 'feature_mimicking':
#             processed_obs = self._preproc_obs(obs['obs'])
#             self.model.eval()
#             input_dict = {
#                 'is_train': False,
#                 'prev_actions': None, 
#                 'obs' : processed_obs,
#                 'rnn_states' : self.rnn_states
#             }

#             with torch.no_grad():
#                 res_dict = self.model(input_dict)
#                 if self.has_central_value:
#                     states = obs['states']
#                     input_dict = {
#                         'is_train': False,
#                         'states' : states,
#                     }
#                     value = self.get_central_value(input_dict)
#                     res_dict['values'] = value
#             return res_dict
#         else:
#             return super().get_action_values(obs)


# class KDAgent(AgentWithTeacher, DAgger, FeatureMimicking):
#     """Knowledge Distillation (KD) agent."""
#     def __init__(self, base_name, params) -> None:
#         super().__init__(base_name, params)
#         self.st_cfg = params['student_teacher']

#         if self.kd_cfg['distillation_mode'] == 'feature_mimicking':
#             if self.kd_cfg['feature_type'] == 'actions':
#                 output_dim = self.env_info['action_space'].shape[0]
#             elif self.kd_cfg['feature_type'] == 'activations':
#                 assert False
#             else:
#                 assert False
#             imitation_head_modules = []
#             for layer, output_size in enumerate(self.kd_cfg['imitation_head_units']):
#                 input_size = self.env_info['observation_space'].shape[0] if layer == 0 else self.kd_cfg['imitation_head_units'][layer - 1]
#                 imitation_head_modules.append(nn.Linear(input_size, output_size))
#                 imitation_head_modules.append(nn.ReLU())
#             imitation_head_modules.append(nn.Linear(self.kd_cfg['imitation_head_units'][-1], output_dim))
#             imitation_head_modules.append(nn.Tanh())
#             self.imitation_head = nn.Sequential(*tuple(imitation_head_modules)).to(self.device)
#             self.imitation_optimizer = optim.Adam(self.imitation_head.parameters(), float(self.kd_cfg['imitation_head_lr']))

#         elif self.kd_cfg['distillation_mode'] == 'dagger':
#             self.imitation_optimizer = optim.Adam(self.model.parameters(), float(self.kd_cfg['imitation_head_lr']))

#         else:
#             pass

#         self.train_result = (torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), self.last_lr, torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]))
#         self.kd_results = defaultdict(list)

#     def calc_gradients(self, input_dict) -> None:
#         """Calculates losses and updates agent parameters."""

#         if self.kd_cfg['distillation_mode'] == 'none':
#             return super().calc_gradients(input_dict)
        
#         else:
#             return getattr(self, 'calc_' + self.kd_cfg['distillation_mode'] + '_gradients')()

    
    
#     def train(self) -> Tuple[float, int]:
#         """Runs traning."""
#         self.init_tensors()
#         self.last_mean_rewards = -100500
#         start_time = time.time()
#         total_time = 0
#         rep_count = 0
#         self.obs = self.env_reset()
#         self.curr_frames = self.batch_size_envs

#         if self.multi_gpu:
#             print("====================broadcasting parameters")
#             model_params = [self.model.state_dict()]
#             dist.broadcast_object_list(model_params, 0)
#             self.model.load_state_dict(model_params[0])

#         while True:
#             epoch_num = self.update_epoch()
#             self.epoch_num = epoch_num
#             step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
#             total_time += sum_time
#             frame = self.frame // self.num_agents

#             # cleaning memory to optimize space
#             self.dataset.update_values_dict(None)
#             should_exit = False

#             if self.rank == 0:
#                 self.diagnostics.epoch(self, current_epoch=epoch_num)
#                 # do we need scaled_time?
#                 scaled_time = self.num_agents * sum_time
#                 scaled_play_time = self.num_agents * play_time
#                 curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
#                 self.frame += curr_frames

#                 if self.print_stats:
#                     step_time = max(step_time, 1e-6)
#                     fps_step = curr_frames / step_time
#                     fps_step_inference = curr_frames / scaled_play_time
#                     fps_total = curr_frames / scaled_time
#                     print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.max_epochs}')

#                 self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
#                 if len(b_losses) > 0:
#                     self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

#                 if self.has_soft_aug:
#                     self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

#                 for k, v in self.kd_results.items():
#                     self.writer.add_scalar(k, sum(v) / len(v), frame)
#                 self.kd_results = defaultdict(list)  # Reset kd_results
#                 self.writer.add_scalar('info/last_teacher_action_prob', self.teacher_action_prob, frame)
#                 teacher_replay_size = self.teacher_dataset.capacity if self.teacher_dataset.full else self.teacher_dataset.idx
#                 self.writer.add_scalar('info/teacher_replay_size', teacher_replay_size, frame)
                
#                 if self.game_rewards.current_size > 0:
#                     mean_rewards = self.game_rewards.get_mean()
#                     mean_lengths = self.game_lengths.get_mean()
#                     self.mean_rewards = mean_rewards[0]

#                     for i in range(self.value_size):
#                         rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
#                         self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
#                         self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
#                         self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

#                     self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
#                     self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
#                     self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

#                     if self.has_self_play_config:
#                         self.self_play_manager.update(self)

#                     checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

#                     if self.save_freq > 0:
#                         if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
#                             self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

#                     if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
#                         print('saving next best rewards: ', mean_rewards)
#                         self.last_mean_rewards = mean_rewards[0]
#                         self.save(os.path.join(self.nn_dir, self.config['name']))

#                         if 'score_to_win' in self.config:
#                             if self.last_mean_rewards > self.config['score_to_win']:
#                                 print('Network won!')
#                                 self.save(os.path.join(self.nn_dir, checkpoint_name))
#                                 should_exit = True

#                 if epoch_num >= self.max_epochs:
#                     if self.game_rewards.current_size == 0:
#                         print('WARNING: Max epochs reached before any env terminated at least once')
#                         mean_rewards = -np.inf
#                     self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + 'ep' + str(epoch_num) + 'rew' + str(mean_rewards)))
#                     print('MAX EPOCHS NUM!')
#                     should_exit = True

#                 update_time = 0

#             if self.multi_gpu:
#                 should_exit_t = torch.tensor(should_exit, device=self.device).float()
#                 dist.broadcast(should_exit_t, 0)
#                 should_exit = should_exit_t.float().item()
#             if should_exit:
#                 return self.last_mean_rewards, epoch_num

#             if should_exit:
#                 return self.last_mean_rewards, epoch_num
