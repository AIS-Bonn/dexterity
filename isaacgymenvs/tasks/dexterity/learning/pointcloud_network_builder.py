import matplotlib.pyplot as plt
from rl_games.algos_torch.network_builder import A2CBuilder
import torch
import torch.nn as nn


class A2CPointcloudBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        self.pointcloud_emb_shape = 64
        mlp_input_dim = 0
        for obs, start_end in kwargs['observation_start_end'].items():
            if 'pointcloud' in obs:
                mlp_input_dim += self.pointcloud_emb_shape
            else:
                mlp_input_dim += start_end[1] - start_end[0]

        kwargs['input_shape'] = (mlp_input_dim,)
        net = A2CPointcloudBuilder.Network(self.params, **kwargs)
        net.observation_start_end = kwargs['observation_start_end']
        return net

    class Network(A2CBuilder.Network):
        def embed_pointcloud(self, name, pointcloud):
            # point_cloud.shape == [batch_size, max_num_points_padded, 4], where the last dim is (x, y, z, mask)
            if not hasattr(self, name + '_encoder'):
                self._build_pointnet(name)
            embedding = getattr(self, name + '_encoder').to(pointcloud.device)(pointcloud.transpose(1, 2)).squeeze(2)
            return embedding

        def _build_pointnet(self, name: str, units=(64, 64, 64), max_num_points_padded=128):
            """Builds a simple PointNet encoder that processes pointclouds of shape 
            [batch_size, max_num_points_padded, 4]. Permutation invariance is created by 
            MaxPood1d at the end."""

            pointnet_modules = []
            for layer, unit in enumerate(units):
                input_size = 4 if layer == 0 else units[layer - 1]
                output_size = unit
                pointnet_modules.append(nn.Conv1d(input_size, output_size, 1))
                pointnet_modules.append(nn.BatchNorm1d(output_size))
                if layer < len(units) - 1:
                    pointnet_modules.append(nn.ReLU())
            pointnet_modules.append(nn.MaxPool1d(max_num_points_padded))
            setattr(self, name + '_encoder', nn.Sequential(*tuple(pointnet_modules)))
            print(f"Encoder for '{name}':", getattr(self, name + '_encoder'))

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
                ax.set_box_aspect((1, 1, 1))
                plt.show()

        def forward(self, obs_dict, show_pointclouds: bool = False):
            """Forward pass of the network. Checks if pointclouds are present in the observation and embeds them 
            before passing the other observations and embedded pointclouds to the regular MPL forward pass of 
            the A2CBuilder.Network."""

            # If point-clouds are present in the observation, embed them before feeding the observation to the MLP.
            if any('pointcloud' in k for k in self.observation_start_end.keys()):
                mlp_input_obs = []
                for obs_name, start_end in self.observation_start_end.items():
                    obs = obs_dict['obs'][:, start_end[0]:start_end[1]]
                    # Embed pointcloud observations.
                    if 'pointcloud' in obs_name:
                        pointcloud = obs.view(obs_dict['obs'].shape[0], -1, 4)
                        if show_pointclouds:
                            self._show_pointcloud(obs_name, pointcloud)
                        obs = self.embed_pointcloud(obs_name, pointcloud)
                    mlp_input_obs.append(obs)
                obs_dict['obs'] = torch.cat(mlp_input_obs, dim=1)
            return super().forward(obs_dict)
