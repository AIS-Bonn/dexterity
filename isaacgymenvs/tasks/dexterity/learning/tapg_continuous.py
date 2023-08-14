from rl_games.algos_torch.a2c_continuous import A2CAgent
from isaacgymenvs.tasks.dexterity.learning.dictobs_continuous import A2CDictObsAgent
from isaacgymenvs.tasks.dexterity.learning.dictobs_player import PpoDictObsPlayerContinuous
from rl_games.common.experience import ExperienceBuffer
import torch
import time
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.common import a2c_common, datasets
from torch import optim
from rl_games.common import common_losses
from rl_games.algos_torch import torch_ext
from rl_games.common.a2c_common import A2CBase
import os
import gym
import numpy as np
from typing import *


class TeacherDataset:
    def __init__(self, capacity: int) -> None:
        super().__init__()
        self._full = False
        self._idx = 0
        self._capacity = int(capacity)
        self._storage_initialized = False

    def __len__(self) -> int:
        return self._capacity if self._full else self._idx
    
    def add(self, data: Tuple[torch.Tensor, ...]) -> None:
        if not self._storage_initialized:
            self._initialize_storage(data)
        assert all(d.shape[0] == data[0].shape[0] for d in data), "Data must have uniform batch_size."
        assert all(d.shape[1:] == s.shape[1:] for d, s in zip(data, self._storage)), "Data shape must match storage."
        assert all(d.dtype == s.dtype for d, s in zip(data, self._storage)), "Data dtype must match storage."
        self._add_to_storage(data)
    
    def _initialize_storage(self, data: Tuple[torch.Tensor, ...]) -> None:
        self._storage = [torch.empty((self._capacity, *d.shape[1:]), dtype=d.dtype, device=d.device) for d in data]
        self._storage_initialized = True

    def _add_to_storage(self, data: Tuple[torch.Tensor, ...]) -> None:
        batch_size = data[0].shape[0]
        remaining_capacity = min(self._capacity - self._idx, batch_size)
        overflow = batch_size - remaining_capacity
        self._full = self._full or remaining_capacity < batch_size
        
        for i, d in enumerate(data):
            # Put overflow data at the beginning of the buffer.
            if remaining_capacity < batch_size:
                self._storage[i][0:overflow] = d[-overflow:]
            self._storage[i][self._idx:self._idx+remaining_capacity] = d[:remaining_capacity]
        self._idx = (self._idx + remaining_capacity) % self._capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        sample_idxs = torch.randint(0, len(self), (batch_size,), device=self._storage[0].device)
        return tuple(d[sample_idxs] for d in self._storage)


class TAPGExperienceBuffer(ExperienceBuffer):
    def __init__(self, env_info, algo_info, device, bc_minibatch_size):
        self.bc_minibatch_size = bc_minibatch_size
        super().__init__(env_info, algo_info, device)
    
    def _init_from_env_info(self, env_info):
        super()._init_from_env_info(env_info)
        bc_base_shape = (self.horizon_length, self.bc_minibatch_size)

        self.tensor_dict['replayed_obses'] = self._create_tensor_from_space(env_info['observation_space'], bc_base_shape)

        val_space = gym.spaces.Box(low=0, high=1,shape=(env_info.get('value_size',1),))
        self.tensor_dict['teacher_values'] = self._create_tensor_from_space(val_space, bc_base_shape)
        self.tensor_dict['replayed_values'] = self._create_tensor_from_space(val_space, bc_base_shape)
        self.tensor_dict['replayed_neglogpacs'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), bc_base_shape)

        if self.is_continuous:
            self.tensor_dict['teacher_actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), bc_base_shape)


class TAPGDataset(datasets.PPODataset):
    def __init__(self, batch_size, minibatch_size, replayed_minibatch_size, is_discrete, is_rnn, device, seq_len):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)
        self.replayed_minibatch_size = replayed_minibatch_size

    def update_replayed_dict(self, replayed_dict):
        self.replayed_dict = replayed_dict

    def _get_item(self, idx):
        start = idx * self.replayed_minibatch_size
        end = (idx + 1) * self.replayed_minibatch_size
        self.last_replayed_range = (start, end)

        replayed_dict = {}
        for k, v in self.replayed_dict.items():
            if k not in self.special_names and v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[start:end] for kd, vd in v.items() }
                    replayed_dict[k] = v_dict
                else:
                    replayed_dict[k] = v[start:end]

        input_dict = super()._get_item(idx)

        return { **input_dict, **replayed_dict }

class TAPGAgent(A2CDictObsAgent):
    '''
    Overall information flow:

    self.play_steps() -> batch_dict

    self.train_epoch()
        self.prepare_dataset(batch_dict)
            self.dataset.update_values_dict(dataset_dict)
        
        for miniep in range(0, self.mini_epochs_num):
            for i in range(len(self.dataset)):
                losses = self.train_actor_critic(dataset[i])
                    self.calc_gradients(input_dict=dataset[i])


    self.experience_buffer is of type rl_games.common.experience.ExperienceBuffer and 
    self.update_list contains ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']

    Which methods need to be changed:
    play_steps to add teacher observations to the experience buffer
    Therefore, a different kind of experience buffer is needed.
    The experience buffer is created in the init_tensors method.
            

    '''

    def __init__(self, base_name, params) -> None:
        self.student_cfg = params['student']
        self.teacher_cfg = params['teacher']
        self._acquire_student(base_name, params)
        self._acquire_teacher(params)

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
            self.dataset = TAPGDataset(self.batch_size, self.minibatch_size, self.teacher_cfg['minibatch_size'], self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)

            if self.normalize_value:
                self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

            self.has_value_loss = self.use_experimental_cv or not self.has_central_value
            self.algo_observer.after_init(self)

    def _acquire_teacher(self, params) -> None:
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
        assert self.teacher_cfg['checkpoint'], "teacher_checkpoint must be specified to run DAgger."

        # Create and restore teacher.
        self.teacher = PpoDictObsPlayerContinuous(teacher_params)
        self.teacher.restore(self.teacher_cfg['checkpoint'])
        self.teacher.has_batch_dimension = True

        self.teacher_dataset = TeacherDataset(capacity=self.teacher_cfg['dataset_size'])

    @property
    def pg_coef_scale_factor(self) -> float:
        return self.student_cfg['pg_coef']['init'] * self.student_cfg['pg_coef']['decay'] ** self.epoch_num
    
    @property
    def bc_coef_scale_factor(self) -> float:
        return self.student_cfg['bc_coef']['init'] * self.student_cfg['bc_coef']['decay'] ** self.epoch_num
    
    def get_pg_coef(self, advantages):
        if self.student_cfg['pg_coef']['type'] == 'advantage':
            return self.pg_coef_scale_factor * advantages
        elif self.student_cfg['pg_coef']['type'] == 'constant':
            return self.pg_coef_scale_factor * torch.ones_like(advantages)
        else:
            assert False
    
    def get_bc_coef(self, teacher_advantages):
        if self.student_cfg['bc_coef']['type'] == 'advantage':
            return self.bc_coef_scale_factor * teacher_advantages  # Weighting to account for different batch sizes.
        elif self.student_cfg['bc_coef']['type'] == 'constant':
            return self.bc_coef_scale_factor * torch.ones_like(teacher_advantages)  # Weighting to account for different batch sizes.
        else:
            assert False

    def init_tensors(self):
        super().init_tensors()
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }

        # Overwrite default experience buffer.
        self.experience_buffer = TAPGExperienceBuffer(self.env_info, algo_info, self.ppo_device, self.teacher_cfg['minibatch_size'] * self.batch_size // (self.minibatch_size * self.horizon_length))
        self.tensor_list += ['teacher_actions', 'teacher_values', 'replayed_obses', 'replayed_neglogpacs', 'replayed_values']

    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        num_envs = self.obs['obs'].shape[0]

        for n in range(self.horizon_length):
            # Query teacher for actions on the current observations and store result.
            teacher_res_dict = self.teacher.model({'is_train': False, 'prev_actions': None, 'obs' : self.obs['teacher_obs'], 'states': self.teacher.states})
            self.teacher_dataset.add((self.obs['obs'], teacher_res_dict['actions'], teacher_res_dict['values']))

            # Sample observations and teacher actions from the replay buffer (aggregated data instead of on-policy data).
            replayed_obs, replayed_teacher_action, replayed_teacher_values = self.teacher_dataset.sample(batch_size=self.teacher_cfg['minibatch_size'] * self.batch_size // (self.minibatch_size * self.horizon_length))

            # Concatenate current and sampled observations to only query the policy once.
            all_obs = {'obs': torch.cat((self.obs['obs'], replayed_obs), dim=0)}

            if self.use_action_masks:
                assert False
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(all_obs)

            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('replayed_obses', n, replayed_obs)
            self.experience_buffer.update_data('dones', n, self.dones)

            self.experience_buffer.update_data('teacher_actions', n, replayed_teacher_action)  # Added to store teacher actions.
            self.experience_buffer.update_data('teacher_values', n, replayed_teacher_values)  # Added to store teacher values.

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k][:num_envs]) # Store results of on-policy data.
                if 'replayed_' + k in self.tensor_list:
                    self.experience_buffer.update_data('replayed_' + k, n, res_dict[k][num_envs:])  # Store results of replayed data.

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'][:num_envs])
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'][:num_envs] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

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
    
    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        teacher_actions = batch_dict['teacher_actions']
        teacher_values = batch_dict['teacher_values']

        replayed_obses = batch_dict['replayed_obses']
        replayed_neglogpacs = batch_dict['replayed_neglogpacs']
        replayed_values = batch_dict['replayed_values']


        advantages = returns - values

        replayed_advantages = torch.clamp(teacher_values - replayed_values, min=0)
        if self.teacher_cfg['gated_advantage']:
            replayed_advantages = (replayed_advantages > 0).float()

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        if self.normalize_advantage:
            if self.is_rnn or self.normalize_rms_advantage:
                assert False

            if self.teacher_cfg['normalize_advantages_together']:
                all_advantages = torch.cat([advantages, replayed_advantages], dim=0)
                all_advantages = torch.sum(all_advantages, axis=1)
                all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
                advantages = all_advantages[:advantages.shape[0]]
                replayed_advantages = all_advantages[advantages.shape[0]:]

            else:
                advantages = torch.sum(advantages, axis=1)
                replayed_advantages = torch.sum(replayed_advantages, axis=1)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                replayed_advantages = (replayed_advantages - replayed_advantages.mean()) / (replayed_advantages.std() + 1e-8)

        # Update advantages based on bc_coef and pg_coef
        advantages = self.get_pg_coef(advantages)
        replayed_advantages = self.get_bc_coef(replayed_advantages)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        replayed_dict = {}
        replayed_dict['teacher_actions'] = teacher_actions
        replayed_dict['old_teacher_values'] = teacher_values
        replayed_dict['old_replayed_values'] = replayed_values

        replayed_dict['replayed_obses'] = replayed_obses
        replayed_dict['replayed_old_logp_actions'] = replayed_neglogpacs
        replayed_dict['replayed_advantages'] = replayed_advantages


        self.dataset.update_values_dict(dataset_dict)
        self.dataset.update_replayed_dict(replayed_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['dones'] = dones
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        
        teacher_actions = input_dict['teacher_actions']
        teacher_value_preds_batch = input_dict['old_teacher_values']

        replayed_obs_batch = input_dict['replayed_obses']
        replayed_obs_batch = self._preproc_obs(replayed_obs_batch)
        replayed_old_action_log_probs_batch = input_dict['replayed_old_logp_actions']
        replayed_advantage = input_dict['replayed_advantages']
        replayed_value_preds_batch = input_dict['old_replayed_values']

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': torch.cat([actions_batch, teacher_actions], dim=0),
            'obs' : torch.cat([obs_batch, replayed_obs_batch], dim=0)
        }

        rnn_masks = None
        if self.is_rnn:
            assert False
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values'][0:obs_batch.shape[0]]
            entropy = res_dict['entropy'][0:obs_batch.shape[0]]
            mu = res_dict['mus'][0:obs_batch.shape[0]]
            sigma = res_dict['sigmas'][0:obs_batch.shape[0]]

            pg_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs[:obs_batch.shape[0]], advantage, self.ppo, curr_e_clip)
            bc_loss = self.actor_loss_func(replayed_old_action_log_probs_batch, action_log_probs[obs_batch.shape[0]:], replayed_advantage, self.student_cfg['ppo'], self.student_cfg['e_clip'])

            # Compute distance between student and teacher actions.
            student_teacher_action_kl = torch.Tensor([0.]) #torch_ext.policy_kl(mu.detach(), sigma.detach(), teacher_mu.detach(), teacher_sigma.detach(), reduce=True)
            student_teacher_action_mse = torch.nn.functional.mse_loss(res_dict['mus'][obs_batch.shape[0]:].detach(), teacher_actions.detach())

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
            losses, sum_mask = torch_ext.apply_masks([pg_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            pg_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            a_loss = pg_loss + torch.mean(bc_loss)

            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

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
            'new_neglogp' : action_log_probs[0:obs_batch.shape[0]],
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, student_teacher_action_kl, student_teacher_action_mse, advantage, replayed_advantage, replayed_value_preds_batch.detach().mean(), teacher_value_preds_batch.detach().mean(), self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
    def train_epoch(self):
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
        student_teacher_action_kls = []
        student_teacher_action_mses = []
        advantages = []
        teacher_advantages = []
        values = []
        teacher_values = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, student_teacher_action_kl, student_teacher_action_mse, advantage, teacher_advantage, value, teacher_value, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                student_teacher_action_kls.append(student_teacher_action_kl)
                student_teacher_action_mses.append(student_teacher_action_mse)
                advantages.append(advantage)
                teacher_advantages.append(teacher_advantage)
                values.append(value)
                teacher_values.append(teacher_value)
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

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, student_teacher_action_kls, student_teacher_action_mses, advantages, teacher_advantages, values, teacher_values, last_lr, lr_mul

    def train(self):
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
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, student_teacher_action_kls, student_teacher_action_mses, advantages, teacher_advantages, values, teacher_values, last_lr, lr_mul = self.train_epoch()
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

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, student_teacher_action_kls, student_teacher_action_mses, advantages, teacher_advantages, values, teacher_values, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

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
            
    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, student_teacher_action_kls, student_teacher_action_mses, advantages, teacher_advantages, values, teacher_values, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        self.writer.add_scalar('info/student_teacher_action_kl', torch_ext.mean_list(student_teacher_action_kls).item(), frame)
        self.writer.add_scalar('info/student_teacher_action_mse', torch_ext.mean_list(student_teacher_action_mses).item(), frame)
        self.writer.add_scalar('info/advantage', torch_ext.mean_list(advantages).item(), frame)
        self.writer.add_scalar('info/teacher_advantage', torch_ext.mean_list(teacher_advantages).item(), frame)
        self.writer.add_scalar('info/value', torch_ext.mean_list(values).item(), frame)
        self.writer.add_scalar('info/teacher_value', torch_ext.mean_list(teacher_values).item(), frame)
        self.writer.add_scalar('info/bc_coef_scale_factor', self.bc_coef_scale_factor, frame)
        self.writer.add_scalar('info/pg_coef_scale_factor', self.pg_coef_scale_factor, frame)
        self.writer.add_scalar('info/teacher_dataset_size', len(self.teacher_dataset), frame)
        super().write_stats(total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
        