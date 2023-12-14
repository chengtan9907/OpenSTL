import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from openstl.modules import RRDB, ResBlock, Up, PredictModel
from openstl.utils import build_similarity_matrix, sim_matrix_interpolate, sim_matrix_postprocess, cum_multiply

class MMVP_Model(nn.Module):
    r"""MMVP

    Implementation of `MMVP: Motion-Matrix-based Video Prediction
    <https://arxiv.org/abs/2308.16154>`_.

    """ 

    def __init__(self, in_shape, aft_seq_length=10,
                 hid_S=32, hid_T=192,
                 rrdb_encoder_num=2, rrdb_decoder_num=2,
                 rrdb_enhance_num=2, downsample_setting='2,2,2',
                 shuffle_setting=True, use_direct_predictor=True,
                 **kwargs):
        super(MMVP_Model, self).__init__()
        T, C, H, W = in_shape
        downsample_ratio = [int(x) for x in downsample_setting.split(',')]
        highres_scale = np.prod(downsample_ratio[:-1]) * 2
        lowres_scale = np.prod(downsample_ratio) * 2
        self.pre_seq_length = T
        self.mat_size = [[H // highres_scale, W // highres_scale], [H // lowres_scale, W // lowres_scale]]
        
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)
        self.enc = RRDBEncoder(C=C, hid_S=hid_S, rrdb_encoder_num=rrdb_encoder_num, downsample_ratio=downsample_ratio)
        self.filter = filter_block(downsample_scale=downsample_ratio, hid_S=hid_S, mat_size=self.mat_size)
        self.dec = RRDBDecoder(C=C, hid_S=hid_S, rrdb_decoder_num=rrdb_decoder_num, downsample_scale=downsample_ratio)
        self.fuse = Compose(downsample_scale=downsample_ratio, mat_size=self.mat_size, 
                            prev_len=T, aft_seq_length=aft_seq_length)

        self.hid = MidMotionMatrix(T=T, hid_S=hid_S, hid_T=hid_T, mat_size=self.mat_size, 
                                   aft_seq_length=aft_seq_length, use_direct_predictor=use_direct_predictor)

        res_shuffle_scale = 1
        for s in range(len(downsample_ratio) - 1):
            res_shuffle_scale *= downsample_ratio[s]
        self.res_shuffle = nn.PixelShuffle(res_shuffle_scale)
        self.res_unshuffle = nn.PixelUnshuffle(res_shuffle_scale)
        self.res_shuffle_scale = res_shuffle_scale

        self.enhance = ImageEnhancer(C_in=C, hid_S=hid_S, downsample_scale=downsample_ratio, rrdb_enhance_num=rrdb_enhance_num)
    
    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x_raw = x_raw.reshape(B*T, C, H, W)
        x_raw = self.unshuffle(x_raw)
        x = x_raw.clone()
        x_wh = x.shape[-2:]
        # encoder
        fi = self.enc(x)  #N, C, H, W

        #record fi shape for later decoder
        feat_shape = []    
        for i in range(len(fi)):
            if fi[i] is None:
                feat_shape.append(None)
            else:
                feat_shape.append(fi[i].shape[2:])
        #filter block
        gi = self.filter(fi)

        #construct and predict similarity matrix
        similarity_matrix = self.hid(gi, B, T)

        # compose motion matrix and embed feature
        composed_fut_feat = self.fuse(fi, similarity_matrix, feat_shape)

        #decoder feature
        recon_img = self.dec(composed_fut_feat)
        final_recon_img = recon_img.clone()

        if x_wh != recon_img.shape[2:]:
            std_w = int(self.mat_size[0][0] * self.res_shuffle_scale)
            std_h = int(self.mat_size[0][1] * self.res_shuffle_scale)
            x_raw = F.interpolate(x_raw, (std_w, std_h))

        image_list = [self.res_unshuffle(x_raw)]
        compose_image, avg_image = self.fuse.feat_compose(image_list, [similarity_matrix[0]])
        compose_image = compose_image[0]
        
        compose_image = self.res_shuffle(compose_image)
        fut_img_seq = self.shuffle(compose_image)

        recon_img = self.shuffle(recon_img)
        final_recon_img = self.shuffle(final_recon_img)
        if fut_img_seq.shape[2:] != final_recon_img.shape[2:]:
            fut_img_seq = F.interpolate(fut_img_seq, final_recon_img.shape[2:])
        final_recon_img = self.enhance(torch.cat([final_recon_img, fut_img_seq], dim=1))

        if recon_img.shape[-2] != H or recon_img.shape[-1] != W:
            recon_img = F.interpolate(recon_img, (H, W))
            final_recon_img = F.interpolate(final_recon_img, (H, W))
        
        recon_img = recon_img.permute(0, 2, 3, 1).reshape(B, -1, C, H, W)
        final_recon_img = final_recon_img.permute(0, 2, 3, 1).reshape(B, -1, C, H, W)

        return final_recon_img
        
        
        
class RRDBEncoder(nn.Module):
    def __init__(self, C=1, hid_S=32, downsample_ratio=[2, 2, 2], rrdb_encoder_num=2, scale_in_use=3):
        super(RRDBEncoder, self).__init__()
        self.C_in = C * 4
        self.hid_S = hid_S
        self.scale_num = len(downsample_ratio)
        self.downsample_ratio = downsample_ratio
        self.scale_in_use = scale_in_use
        self.inconv = nn.Conv2d(self.C_in, self.hid_S, 3, 1, 1)
        self.block_rrdb = nn.Sequential(*[RRDB(hid_S) for i in range(rrdb_encoder_num)])

        pre_downsample_block_list = []
        
        for i in range(self.scale_num-2):
            pre_downsample_block_list.append(ResBlock(hid_S * (2 ** i), hid_S * (2 ** (i+1)), 
                                                      downsample=True, factor=downsample_ratio[i]))
        self.pre_downsample_block = nn.ModuleList(pre_downsample_block_list)
        
        self.downsample_high = ResBlock(hid_S * ( 2 ** (self.scale_num-2)), hid_S * ( 2 ** (self.scale_num-1)),
                                        downsample=True, factor=downsample_ratio[-2])
        self.downsample_low = ResBlock(hid_S * (2 ** (self.scale_num-1)), hid_S * (2 ** (self.scale_num)),
                                       downsample=True, factor=downsample_ratio[-1])


    def forward(self, x, save_all=False):
        in_feat = []
        x = self.inconv(x)
        x = self.block_rrdb(x)
        in_feat.append(x)
        for i in range(self.scale_num-2):
            x = self.pre_downsample_block[i](x) 
            in_feat.append(x)
        x = self.downsample_high(x)
        in_feat.append(x)
        x = self.downsample_low(x)
        in_feat.append(x)
        if self.scale_in_use == 3:
            for i in range(len(in_feat) - 3):
                in_feat[i] = None
        elif self.scale_in_use == 2:
            for i in range(len(in_feat)-2):
                in_feat[i] = None
        return in_feat

class filter_block(nn.Module):
    def __init__(self, downsample_scale, hid_S, mat_size):
        super(filter_block, self).__init__()
        self.filter_block = []
        high_scale = len(downsample_scale) - 1
        feat_len = hid_S * (2 ** high_scale)
        self.mat_size = mat_size
        self.filter_block.append(nn.Sequential(nn.Conv2d(feat_len, hid_S, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S),
                                               nn.LeakyReLU()))
        low_scale = high_scale + 1
        feat_len = hid_S * (2 ** low_scale)
        self.filter_block.append(nn.Sequential(nn.Conv2d(feat_len, hid_S * 2, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S * 2),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S * 2, hid_S * 2, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S * 2),
                                               nn.LeakyReLU(),
                                               nn.Conv2d(hid_S * 2, hid_S * 2, kernel_size=3, padding=1),
                                               nn.BatchNorm2d(hid_S * 2),
                                               nn.LeakyReLU()))
    
        self.filter_block = nn.ModuleList(self.filter_block)

    def forward(self, x):
        gi = []
        for s in [-2, -1]:
            feat_area = x[s].shape[-1] * x[s].shape[-2]
            mat_area = self.mat_size[s][-1] * self.mat_size[s][-2]
            if mat_area != feat_area:
                out = F.interpolate(x[s].clone(), size=tuple(self.mat_size[s]), mode='bilinear')
            else:
                out = x[s].clone()
            out = self.filter_block[s](out)
            gi.append(out)
        return gi

        

class MidMotionMatrix(nn.Module):
    def __init__(self, T, hid_S=32, hid_T=192, mat_size=[[8, 8], [4, 4]], 
                 aft_seq_length=10, use_direct_predictor=True):
        super(MidMotionMatrix, self).__init__()
        self.pre_seq_len = T
        self.mat_size = mat_size
        self.mx_h = mat_size[0][0]
        self.mx_w = mat_size[0][1]
        self.scale_fuser_1 = Up(hid_S * 2, hid_S, bilinear=False, scale=2)
        self.scale_fuser_2 = nn.Sequential(nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(hid_S),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(hid_S, hid_S, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(hid_S),
                                           nn.LeakyReLU())
        self.predictor = PredictModel(T=T, hidden_len=hid_T, aft_seq_length=aft_seq_length,
                                      mx_h=self.mx_h, mx_w=self.mx_w,
                                      use_direct_predictor=use_direct_predictor)

    def forward(self, x, B, T):
        similar_matrix = []
        prev_sim_matrix = []
        pred_sim_matrix = [None, None]
        # construct similarity matrix
        for i in [-2, -1]:
            N = x[i].shape[0]
            h, w = x[i].shape[2:]
            cur_sim_matrix = build_similarity_matrix(x[i].reshape(B, T, -1, h, w))
            prev_sim_matrix.append(cur_sim_matrix[:, :self.pre_seq_len-1].clone())
        
        pred_fut_matrix, _ = self.predictor(prev_sim_matrix[0], softmax=False, res=None)
        pred_sim_matrix[0] = pred_fut_matrix.clone()
        pred_sim_matrix[1] = sim_matrix_interpolate(pred_fut_matrix.clone(), self.mat_size[0], self.mat_size[1])
        # post process the matrix
        pred_sim_matrix[0] = sim_matrix_postprocess(pred_sim_matrix[0])
        pred_sim_matrix[1] = sim_matrix_postprocess(pred_sim_matrix[1])

        #update similarity matrix list
        for i in range(len(prev_sim_matrix)):
            new_cur_sim_matrix = torch.cat([sim_matrix_postprocess(prev_sim_matrix[i]), pred_sim_matrix[i]], dim=1)
            similar_matrix.append(new_cur_sim_matrix)
        return similar_matrix

class Compose(nn.Module):
    def __init__(self, downsample_scale, mat_size, prev_len, aft_seq_length):
        super(Compose, self).__init__()
        self.downsample_scale = downsample_scale
        self.mat_size = mat_size
        self.prev_len = prev_len
        self.aft_seq_length = aft_seq_length
        self.feat_shuffle = []
        self.feat_unshuffle = []
        self.feat_scale_list = []
        for i in range(len(self.downsample_scale) - 1):
            feat_shuffle_scale = 1
            for s in range(len(self.downsample_scale) - 2, i - 1, -1):
                feat_shuffle_scale *= self.downsample_scale[s]
            self.feat_scale_list.append(feat_shuffle_scale)
            self.feat_shuffle.append(nn.PixelShuffle(feat_shuffle_scale))
            self.feat_unshuffle.append(nn.PixelUnshuffle(feat_shuffle_scale))
        self.feat_shuffle = nn.ModuleList(self.feat_shuffle)
        self.feat_unshuffle = nn.ModuleList(self.feat_unshuffle)

    def feat_generator(self, feats, sim_matrix, feat_idx, img_compose=False, scale=1):
        '''

        :param feats: [B,T,c,h,w]
        :param sim_matrix: [B,T,h*w,h*w]
        :return: new_feats: [B,c,h,w]
        '''
        B, T, c, h, w = feats.shape
        # only test single motion
        if scale > 1: # if hw_cur != hw_target, only use the last sim matrix
            feats = feats[:,-1:,]
            sim_matrix = sim_matrix[:,-1:]
            T = 1
        feats = feats.permute(0, 2, 1, 3, 4)  # (B,c,T,h,w)
        feats = feats.reshape(B, c, T * h * w).permute(0, 2, 1)  # (B,Prev T*h*w,c)
        B,T,hw_cur,hw_target = sim_matrix.shape
        sim_matrix = sim_matrix.reshape(B, T * hw_cur, hw_target).permute(0, 2, 1) # Batch, fut H*W, Prev T*HW
        weight = torch.sum(sim_matrix, dim=-1).reshape(-1, 1, hw_target) + 1e-6
        new_feats = torch.bmm(sim_matrix, feats).permute(0, 2, 1) / weight
        new_feats = new_feats.reshape(B, c, h*scale, w*scale)

        return new_feats

    def feat_compose(self, emb_feat_list, sim_matrix, img_compose=False, scale=1, use_gt=False):
        '''

        :param emb_feat_list: (scale_num, (B,T,c,h,w))
        :param sim_matrix:  (B,T-1,h,w,h,w)
        :param use_gt_sim_matrix: bool
        :return: fut_emb_feat_list (scale_num, (B,t,c,h,w))
        '''
        fut_emb_feat_list = []
        ori_emb_feat_list = []
        for i in range(len(emb_feat_list)):
            if emb_feat_list[i] is None:
                fut_emb_feat_list.append(None)
                ori_emb_feat_list.append(None)
                continue

            fut_emb_feat_list.append([])
            cur_emb_feat = emb_feat_list[i]
            ori_emb_feat_list.append(torch.mean(emb_feat_list[i], dim=1))
            
            sim_matrix_seq = sim_matrix[i]
            B = sim_matrix_seq.shape[0]
            N, c, h, w = cur_emb_feat.shape
            cur_emb_feat = cur_emb_feat.reshape(B,-1,c,h,w)
            cur_emb_feat = cur_emb_feat[:,:self.prev_len] if (not use_gt) else cur_emb_feat.clone()

            for t in range(self.aft_seq_length):
                active_matrix_seq = sim_matrix_seq[:,:(self.prev_len-1)]
                if t > 0:
                    fut_t_matrix =sim_matrix_seq[:,(self.prev_len+t-1):(self.prev_len+t)]
                else:
                    fut_t_matrix = sim_matrix_seq[:,(self.prev_len-1):(self.prev_len+t)]
                active_matrix_seq = torch.cat([active_matrix_seq,fut_t_matrix],dim=1)
            
                cur_sim_matrix = cum_multiply(active_matrix_seq.clone())  # B, T+1, h,w,h,w
                composed_t_feats = self.feat_generator(cur_emb_feat[:, :self.prev_len].clone(),
                                                        cur_sim_matrix,feat_idx=i,img_compose=img_compose,scale=scale)
                                                    
                fut_emb_feat_list[i].append(composed_t_feats.clone())
                # update future frame features in the emb_feat_list
                if (not use_gt):
                    if scale == 1:
                        if  cur_emb_feat.shape[1] > self.prev_len+t:
                            cur_emb_feat[:,t+self.prev_len] = composed_t_feats.clone()
                        else:
                            cur_emb_feat = torch.cat([cur_emb_feat,composed_t_feats.clone().unsqueeze(1)],dim=1) #cat compose features for next frame prediction

            temp = torch.stack(fut_emb_feat_list[i], dim=1)
            
            fut_emb_feat_list[i] = temp.reshape(-1, c, h*scale, w*scale) # B*T,c,h,w

        return fut_emb_feat_list,ori_emb_feat_list

    def forward(self, x, similar_matrix, feat_shape):
        compose_feat_list = []
        similar_matrix_for_compose = []
        for i in range(len(x)):
            if x[i] is None:
                compose_feat_list.append(None)
                similar_matrix_for_compose.append(None)
                continue
            if i < len(x) - 2:
                h, w = x[i].shape[-2:]
                target_size = (h // self.feat_scale_list[i] * self.feat_scale_list[i], w // self.feat_scale_list[i] * self.feat_scale_list[i])
                cur_feat = self.feat_unshuffle[i](F.interpolate(x[i].clone(), size=target_size, mode='bilinear'))

                if (cur_feat.shape[-2] != self.mat_size[0][-2]) or (cur_feat.shape[-1] != self.mat_size[0][-1]):
                    compose_feat_list.append(F.interpolate(cur_feat, size=tuple(self.mat_size[0]), mode='bilinear'))
                else:
                    compose_feat_list.append(cur_feat.clone())
                
                similar_matrix_for_compose.append(similar_matrix[0])
            else:
                if (x[i].shape[-2] != self.mat_size[i - len(x) + 2][-2]) or (x[i].shape[-1] != self.mat_size[i - len(x) + 2][-1]):
                    compose_feat_list.append(F.interpolate(x[i], size=tuple(self.mat_size[i - len(x) + 2]), mode='bilinear'))
                else:
                    compose_feat_list.append(x[i])
        
        similar_matrix_for_compose.append(similar_matrix[0])
        similar_matrix_for_compose.append(similar_matrix[1])

        compose_fut_feat_list, _ = self.feat_compose(compose_feat_list, similar_matrix_for_compose)

        for i in range(len(compose_fut_feat_list)):
            if compose_fut_feat_list[i] is None:
                continue
            if i < len(x) - 2:
                compose_fut_feat_list[i] = self.feat_shuffle[i](compose_fut_feat_list[i])
            if (compose_fut_feat_list[i].shape[-2] != feat_shape[i][-2]) or (compose_fut_feat_list[i].shape[-1] != feat_shape[i][-1]):
                compose_fut_feat_list[i] = F.interpolate(compose_fut_feat_list[i], size=tuple(feat_shape[i]), mode='bilinear')

        return compose_fut_feat_list
            


        
class RRDBDecoder(nn.Module):
    def __init__(self, C=1, hid_S=32, downsample_scale=[2,2,2], rrdb_decoder_num=2, scale_in_use=3):
        super(RRDBDecoder, self).__init__()

        self.scale_num = len(downsample_scale)
        out_channel = C * 4 
 
        self.upsample_block_low2high = Up(in_channels=hid_S * (2 ** self.scale_num),
                                          out_channels=hid_S * (2 ** (self.scale_num - 1)),
                                          bilinear=False,
                                          scale=downsample_scale[-1])

        upsample_block_list = []
        for i in range(self.scale_num - 2, -1, -1):
            skip=False if ((i<self.scale_num-1 and scale_in_use == 2) or (i<self.scale_num-2 and scale_in_use == 3)) else True
            upsample_block_list.append(Up(in_channels=hid_S * (2 ** (i+1)),
                                          out_channels=hid_S * (2 ** i),
                                          bilinear=False,
                                          scale=downsample_scale[i],
                                          skip=skip))
        self.upsample_block =  nn.ModuleList(upsample_block_list)

        self.rrdb_block = nn.Sequential(*[RRDB(hid_S) for i in range(rrdb_decoder_num)])

        self.outc = nn.Conv2d(hid_S, out_channel, kernel_size=1)

    def forward(self, in_feat):

        x = self.upsample_block_low2high(in_feat[-1], in_feat[-2])
        for i in range(self.scale_num-1):
            x = self.upsample_block[i](x,in_feat[-i-3])

        x = self.rrdb_block(x)
        logits = self.outc(x)
        return logits

class ImageEnhancer(nn.Module):
    def __init__(self, C_in=1, hid_S=32, downsample_scale=[2,2,2], rrdb_enhance_num=2):
        super(ImageEnhancer, self).__init__()
        self.C_in = C_in
        layers = [nn.Conv2d(C_in * 2, hid_S, 3, 1, 1)]
        for i in range(rrdb_enhance_num):
            layers.append(RRDB(hid_S))
        self.model = nn.Sequential(*layers)

        self.outconv = nn.Conv2d(hid_S, C_in, kernel_size=1)

    def forward(self, x,):
        feat = self.model(x)
        out = self.outconv(feat)
        return out