from rl_games.algos_torch.network_builder import A2CBuilder
import torch
import torch.nn as nn


class AIDBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        self.aid_architecture = 'concatenation'

        self.method = kwargs['agent_params']['student_teacher']['method'] if 'agent_params' in kwargs.keys() else 'none'

        observations_num = kwargs['input_shape'][0]
        actions_num = kwargs['actions_num']

        if self.method == 'aid':
            if self.go_architecture == 'skip_connection':
                kwargs['input_shape'] = (observations_num,)
            elif self.go_architecture == 'concatenation':
                kwargs['input_shape'] = (observations_num + actions_num,)  # Input to the MLP will be observations & estimated teacher actions. 
            else:
                assert False, 'Unknown go_architecture: {}'.format(self.go_architecture)

        net = AIDBuilder.Network(self.params, **kwargs)
        net.st_cfg = kwargs['agent_params']['student_teacher'] if 'agent_params' in kwargs.keys() else None
        net.observations_num = observations_num
        net.actions_num = actions_num
        net.method = self.method
        net.aid_architecture = self.aid_architecture

        if self.method == 'aid':
            net._build_imitation_head()

        return net

    class Network(A2CBuilder.Network):
        def forward(self, obs_dict):
            if self.method == 'aid':
                est_teacher_actions = self.imitation_head(obs_dict['obs']).detach()

                if self.aid_architecture == 'skip_connection':
                    mu, sigma, value, states = super().forward(obs_dict)
                    mu += self.skip_connection(est_teacher_actions)
                    return mu, sigma, value, states
                elif self.aid_architecture == 'concatenation':
                    obs_dict['obs'] = torch.cat((obs_dict['obs'], est_teacher_actions), dim=1)
            return super().forward(obs_dict)

        def _build_imitation_head(self):
            imitation_head_modules = []
            for layer, output_size in enumerate(self.st_cfg['imitation_head_units']):
                input_size = self.observations_num if layer == 0 else self.st_cfg['imitation_head_units'][layer - 1]
                imitation_head_modules.append(nn.Linear(input_size, output_size))
                imitation_head_modules.append(nn.ReLU())
            imitation_head_modules.append(nn.Linear(self.st_cfg['imitation_head_units'][-1], self.actions_num))
            imitation_head_modules.append(nn.Tanh())
            self.imitation_head = nn.Sequential(*tuple(imitation_head_modules))

            if self.aid_architecture == 'skip_connection':
                self.skip_connection = nn.Sequential(
                    nn.Linear(self.actions_num, self.actions_num),
                )
                self.skip_connection[0].weight.data.copy_(torch.eye(self.actions_num))
                self.skip_connection[0].bias.data.copy_(torch.zeros(self.actions_num))
        