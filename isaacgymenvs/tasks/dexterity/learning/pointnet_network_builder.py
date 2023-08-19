import matplotlib.pyplot as plt
from rl_games.algos_torch.network_builder import A2CBuilder
import torch
import torch.nn as nn
from typing import *
from copy import deepcopy


class GlobalMaxPool1d(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
    def forward(self, input):
        return torch.max(input, 2, keepdim=True)[0].view(-1, self.output_dim)


class Mean1d(nn.Module):
    def forward(self, input):
        #print("input.shape:", input.shape)
        return torch.mean(input, 2)



class A2CPointNetBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        pointcloud_emb_shape = 64

        mlp_input_dim = 0
        for obs, start_end in kwargs['observation_start_end'].items():
            if 'pointcloud' in obs:
                mlp_input_dim += pointcloud_emb_shape
            else:
                mlp_input_dim += start_end[1] - start_end[0]

        kwargs['input_shape'] = (mlp_input_dim,)
        net = A2CPointNetBuilder.Network(self.params, **kwargs)
        return net

    class Network(A2CBuilder.Network):

        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            self.use_pointcloud_mask = True

            self._forward_counter = 0

            self.observation_start_end = kwargs['observation_start_end']
            for obs, start_end in kwargs['observation_start_end'].items():
                if 'pointcloud' in obs:
                    self._build_pointnet(obs, has_mask=self.use_pointcloud_mask)

            self.num_pointclouds = sum('pointcloud' in k for k in self.observation_start_end.keys())
            print("num_pointclouds:", self.num_pointclouds)
            if self.num_pointclouds == 1:
                for obs_name, start_end in self.observation_start_end.items():
                    if 'pointcloud' in obs_name:
                        self.pointcloud_start_end = start_end
                        break
            elif self.num_pointclouds > 1:
                assert False

        def embed_pointcloud(self, name, pointcloud):
            # point_cloud.shape == [batch_size, max_num_points_padded, 4], where the last dim is (x, y, z, mask)
            # Transpose (1, 2) in point-cloud to adhere to PyTorch's channel-first convention.
            # print("pointcloud.shape:", pointcloud.shape)

            # print("pointcloud[0, :, 0].mean():", pointcloud[0, :, 0].mean())
            # print("pointcloud[0, :, 1].mean():", pointcloud[0, :, 1].mean())
            # print("pointcloud[0, :, 2].mean():", pointcloud[0, :, 2].mean())

            if self.use_pointcloud_mask:
                input_pointcloud = pointcloud.transpose(1, 2)
            else:
                input_pointcloud = pointcloud[:, :, 0:3].transpose(1, 2)
            embedding = self.pointcloud_encoder(input_pointcloud)
            #embedding = pointcloud[:,:, 0:3].mean(dim=1)
            return embedding

        #def _build_pointnet(self, name: str, conv1d_units=(64, 256, 512), fc_units=(512, 256, 256)):
        def _build_pointnet(self, name: str, conv1d_units=[16, 32, 64], fc_units=[], has_mask=False):
            """Builds a simple PointNet encoder that processes pointclouds of shape 
            [batch_size, max_num_points_padded, 4]. Permutation invariance is created by 
            MaxPood1d at the end."""
            point_features_dim = 4 if has_mask else 3

            pointnet_modules = []
            for layer, unit in enumerate(conv1d_units):
                input_size = point_features_dim if layer == 0 else conv1d_units[layer - 1]
                output_size = unit
                pointnet_modules.append(nn.Conv1d(input_size, output_size, 1))
                pointnet_modules.append(nn.BatchNorm1d(output_size))
                if layer < (len(conv1d_units) + len(fc_units)) - 1:
                    pointnet_modules.append(nn.ReLU())

            #pointnet_modules.append(GlobalMaxPool1d(output_dim=conv1d_units[-1]))
            pointnet_modules.append(Mean1d())

            for layer, unit in enumerate(fc_units):
                input_size = conv1d_units[-1] if layer == 0 else fc_units[layer - 1]
                output_size = unit
                pointnet_modules.append(nn.Linear(input_size, output_size))

                #if layer == len(fc_units) - 2:
                #    pointnet_modules.append(nn.Dropout(p=0.3))

                pointnet_modules.append(nn.BatchNorm1d(output_size))

                if layer < len(fc_units) - 1:
                    pointnet_modules.append(nn.ReLU())

            self.pointcloud_encoder = nn.Sequential(*tuple(pointnet_modules))
            print(f"Encoder for '{name}':", getattr(self, 'pointcloud_encoder'))

        def _show_pointcloud(self, name, pointcloud):
            import matplotlib.pyplot as plt
            for env_id in range(pointcloud.shape[0]):
                # Get unpadded pointcloud.
                unpadded_pointcloud = pointcloud[env_id]
                mask = unpadded_pointcloud[:, 3] > 0
                unpadded_pointcloud = unpadded_pointcloud[mask, 0:3]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(unpadded_pointcloud[:, 0].cpu(), unpadded_pointcloud[:, 1].cpu(), unpadded_pointcloud[:, 2].cpu())
                ax.set_title(f"'{name}' in Environment {env_id}")
                ax.set_xlim3d(-0.07, 0.63)
                ax.set_ylim3d(-0.17, 0.83)
                ax.set_zlim3d(0., 0.25)
                ax.set_box_aspect((0.7, 1, 0.25))
                #ax.auto_scale_xyz([-0.07, 0.63], [-0.17, 0.83], [0.0, 0.25])
                plt.show()

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)
            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                assert False
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)                    

                if self.has_rnn:
                    if not self.is_rnn_before_mlp:
                        a_out_in = a_out
                        c_out_in = c_out
                        a_out = self.actor_mlp(a_out_in)
                        c_out = self.critic_mlp(c_out_in)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_out_in], dim=1)
                            c_out = torch.cat([c_out, c_out_in], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0,1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]                        
                    a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                    c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                    a_out = a_out.transpose(0,1)
                    c_out = c_out.transpose(0,1)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)
                    if self.rnn_ln:
                        a_out = self.a_layer_norm(a_out)
                        c_out = self.c_layer_norm(c_out)
                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)

                if self.num_pointclouds > 0:
                    pointcloud, vector_obs = self._split_obs(obs_dict['obs'])
                    pointcloud_emb = self.embed_pointcloud('pointcloud', pointcloud)
                    out = torch.cat([pointcloud_emb, vector_obs], dim=1)             

                if self.has_rnn:
                    out_in = out
                    if not self.is_rnn_before_mlp:
                        out_in = out
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, out_in], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    out = out.transpose(0, 1)
                    if dones is not None:
                        dones = dones.reshape(num_seqs, seq_length, -1)
                        dones = dones.transpose(0, 1)
                    out, states = self.rnn(out, states, dones, bptt_len)
                    out = out.transpose(0, 1)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)
                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))

                    if "jit_trace" in obs_dict.keys():
                        return [mu, sigma, value]
                    return mu, mu*0 + sigma, value, states
        
        def _split_obs(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
            pointcloud = obs[:, self.pointcloud_start_end[0]:self.pointcloud_start_end[1]].view(obs.shape[0], -1, 4)
            vector_obs = torch.cat([obs[:, 0:self.pointcloud_start_end[0]], obs[:, self.pointcloud_start_end[1]:]], dim=1)
            return pointcloud, vector_obs
        
        # def _embed_pointclouds(self, obs_dict, show_pointclouds_freq: int = None) -> Dict[str, Any]:
        #     if self.num_pointclouds > 0:
        #         pointcloud, vector_obs = self._split_obs(obs_dict['obs'])

        #         if show_pointclouds_freq is not None and self._forward_counter % show_pointclouds_freq == 0:
        #             self._show_pointcloud('pointcloud', pointcloud)
        #             self._forward_counter += 1

        #         pointcloud_emb = self.embed_pointcloud('pointcloud', pointcloud)
        #         obs_dict['obs'] = torch.cat([pointcloud_emb, vector_obs], dim=1)

        #     return obs_dict
            
