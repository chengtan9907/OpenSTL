import math
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

def build_similarity_matrix(emb_feats,thre=-1,sigmoid=False,k=-1,cut_off=False):
    '''

    :param emb_feats: a sequence of embeddings for every frame (N,T,c,h,w)
    :return: similarity matrix (N, T-1, h*w, h*w) current frame --> next frame
    '''
    B,T,c,h,w = emb_feats.shape
    emb_feats = emb_feats.permute(0,1,3,4,2) #  (B,T,h,w,c)
    normalize_feats = emb_feats / (torch.norm(emb_feats,dim=-1,keepdim=True)+1e-6) #  (B,T,h,w,c)
    prev_frame = normalize_feats[:,:T-1].reshape(-1,h*w,c) # (B*(T-1),h*w,c)
    next_frame = normalize_feats[:,1:].reshape(-1,h*w,c,) # (B*(T-1),h*w,c)
    similarity_matrix = torch.bmm(prev_frame,next_frame.permute(0,2,1)).reshape(B,T-1,h*w,h*w) # (N*(T-1)*h*w)
    
    if cut_off:
        similarity_matrix = cut_off_process(similarity_matrix,thre,sigmoid,k)

    return similarity_matrix

def sim_matrix_postprocess(similar_matrix):
    B,T,hw1,hw2 = similar_matrix.shape

    similar_matrix = similar_matrix.reshape(similar_matrix.shape[0],similar_matrix.shape[1],-1)
    similar_matrix = F.softmax(similar_matrix,dim=-1)


    return similar_matrix.reshape(B,T,hw1,hw2)

def sim_matrix_interpolate(in_matrix,ori_hw,target_hw):

    ori_h,ori_w = ori_hw[0],ori_hw[1]
    target_h,target_w = target_hw[0],target_hw[1]
    B,T,hw,hw = in_matrix.shape
    ori_matrix = in_matrix.clone().reshape(B,T,ori_h,ori_w,ori_h,ori_w)
    ori_matrix_half = F.interpolate(ori_matrix.reshape(-1,ori_h,ori_w).unsqueeze(1),(int(target_h),int(target_w)),mode='bilinear').squeeze(1) # (BThw,target_h,target_w)
    new_matrix = F.interpolate(ori_matrix_half.reshape(B,T,ori_h,ori_w,target_h,target_w).permute(0,1,4,5,2,3).reshape(-1,ori_h,ori_w).unsqueeze(1),(int(target_h),int(target_w)),mode='bicubic').squeeze(1) #(BT*targethw,target_h,target_w)
    new_matrix = new_matrix.reshape(B,T,target_h,target_w,target_h,target_w).permute(0,1,4,5,2,3).reshape(B,T,target_h*target_w,target_h*target_w)

    return new_matrix

def cut_off_process(similarity_matrix,thre,sigmoid=False,k=-1):

    B = similarity_matrix.shape[0]
    T_prime = similarity_matrix.shape[1]
    hw = similarity_matrix.shape[2]
    new_similarity_matrix = similarity_matrix.clone()
    #mask all diagonal to zeros
    '''
    diagonal_mask = torch.zeros_like(new_similarity_matrix[0,0]).to(similarity_matrix.device).bool() #(h*w,h*w)
    diagonal_mask.fill_diagonal_(True)
    diagonal_mask = diagonal_mask.reshape(1,1,hw,hw).repeat(B,T_prime,1,1)
    new_similarity_matrix[diagonal_mask] = 0.
    '''
    if sigmoid:
        new_similarity_matrix[new_similarity_matrix<0] = 0.
        new_similarity_matrix = F.sigmoid(new_similarity_matrix)
        #similarity_matrix = F.sigmoid((similarity_matrix+1)/2.)
    elif k > -1: # select top k
        new_similarity_matrix[new_similarity_matrix<0.] = 0.
        select_num = int(new_similarity_matrix.shape[-1] * k)
        top_k,_ = torch.topk(new_similarity_matrix,select_num,dim=-1)
        thre_value = top_k[:,:,:,-1:]
        new_similarity_matrix[new_similarity_matrix<thre_value] = 0.
    else:
        new_similarity_matrix[new_similarity_matrix<thre] = 0.

    return new_similarity_matrix

def cum_multiply(value_seq, cum_softmax = False,reverse=True):
    '''

    :param value_seq: (B,S,***), B - batch num; S- sequence len
    :return: output value_seq: (B,S,***)
    '''
    #print(value_seq.shape)
    if not reverse: # reverse means last element is the one multiplied most times,i.e. the reference is the last element:
        value_seq = torch.flip(value_seq,dims=[1])
    B,T,hw,hw = value_seq.shape
    new_output = value_seq.clone()
    for i in range(value_seq.shape[1]-2,-1,-1):
        cur_sim = new_output[:, i].reshape(B,hw,hw).clone()
        next_sim = new_output[:,i+1].reshape(B,hw,hw).clone()
        new_output[:,i] = torch.bmm(cur_sim,next_sim).reshape(B,hw,hw)
    
    if not reverse:
        new_output = torch.flip(new_output,dims=[1])
    if cum_softmax:
        new_output = sim_matrix_postprocess(new_output)
    return new_output

class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=1e-6, last_epoch=-1,ratio=0.5):
        self.T_period = list(T_period)
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = list(restarts)
        self.restart_weights = [ratio ** (i+1) for i in range(len(restarts))]
        self.last_restart = 0
        print('restart ratio: ',ratio,' T_period: ',T_period,' minimum lr: ',eta_min)
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[list(self.restarts).index(self.last_epoch) + 1]
            weight = self.restart_weights[list(self.restarts).index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]