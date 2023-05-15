import torch
import torch.nn as nn
from openstl.modules import Routing, MVFB, RoundSTE, warp


class DMVFN_Model(nn.Module):
    def __init__(self, in_planes, num_features, configs):
        super(DMVFN_Model, self).__init__()
        self.configs = configs
        self.input_C = configs.in_shape[1]
        self.stu = nn.ModuleList([MVFB(in_planes, num_features[i])
                                  for i in range(configs.num_block)])

        self.routing = Routing(2*self.input_C, configs.routing_out_channels)
        self.l1 = nn.Linear(configs.routing_out_channels, configs.num_block)

    def forward(self, x, training=True):
        batch_size, T, C, height, width = x.shape
        x = x.view(batch_size, T*C, height, width)
        ref = self.get_routing_vector(x)


        img0, img1 = x[:, :C], x[:, C:2*C]
        flow_list, merged_final, mask_final = [], [], []
        warped_img0, warped_img1 = img0, img1

        flow = torch.zeros(batch_size, 4, height, width).to(x.device)
        mask = torch.zeros(batch_size, 1, height, width).to(x.device)

        if training:
            for i in range(self.configs.num_block):
                flow_d, mask_d = self.stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), dim=1), flow,
                                        scale=self.configs.scale[i])

                flow_right_now = flow + flow_d
                mask_right_now = mask + mask_d

                flow = flow + (flow_d) * ref[:, i].reshape(batch_size, 1, 1, 1)
                mask = mask + (mask_d) * ref[:, i].reshape(batch_size, 1, 1, 1)
                flow_list.append(flow)

                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])

                warped_img0_right_now = warp(img0, flow_right_now[:, :2])
                warped_img1_right_now = warp(img1, flow_right_now[:, 2:4])

                if i < self.configs.num_block - 1:
                    mask_final.append(torch.sigmoid(mask_right_now))
                    merged_student_right_now = (warped_img0_right_now, warped_img1_right_now)
                    merged_final.append(merged_student_right_now)
                else:
                    mask_final.append(torch.sigmoid(mask))
                    merged_student = (warped_img0, warped_img1)
                    merged_final.append(merged_student)

            for i in range(self.configs.num_block):
                merged_final[i] = merged_final[i][0] * mask_final[i] + merged_final[i][1] * (1 - mask_final[i])
                merged_final[i] = torch.clamp(merged_final[i], 0, 1)
            return merged_final
        else:
            for i in range(self.configs.num_block):
                if ref[0, i]:
                    flow_d, mask_d = self.stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), dim=1), flow,
                                            scale=self.configs.scale[i])
                    flow = flow + flow_d
                    mask = mask + mask_d

                    mask_final.append(torch.sigmoid(mask))
                    flow_list.append(flow)
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    merged_student = (warped_img0, warped_img1)
                    merged_final.append(merged_student)
            length = len(merged_final)
            for i in range(length):
                merged_final[i] = merged_final[i][0] * mask_final[i] + merged_final[i][1] * (1 - mask_final[i])
                merged_final[i] = torch.clamp(merged_final[i], 0, 1)
            return merged_final

    def get_routing_vector(self, x):
        C = self.input_C
        routing_vector = self.routing(x[:, :2*C]).reshape(x.shape[0], -1)
        routing_vector = torch.sigmoid(self.l1(routing_vector))
        routing_vector = self.configs.beta * self.configs.num_block * \
                         routing_vector / (routing_vector.sum(1, True) + 1e-6)
        routing_vector = torch.clamp(routing_vector, 0, 1)
        ref = RoundSTE.apply(routing_vector)
        return ref