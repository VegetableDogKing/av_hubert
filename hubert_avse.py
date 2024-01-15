import torch
import torch.nn as nn
from argparse import Namespace
# from fairseq import checkpoint_utils, utils
import pdb

class Blstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1,dropout=0):
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout, bidirectional=True)

    def forward(self,x):
        out,_=self.blstm(x)
        out = out[:,:,:int(out.size(-1)/2)]+out[:,:,int(out.size(-1)/2):] 
        return out

    
class BLSTM_WS(nn.Module):
    
    def __init__(self,no_video=False, crossfeat=False):
        super().__init__()

        self.dim = 768
        weight_dim = 12
        self.no_video = no_video
        self.crossfeat = crossfeat
        # if self.weighted_sum:
        self.weight = nn.Parameter(torch.ones(weight_dim))
        self.softmax = nn.Softmax(-1)
        layer_norm  = []
        for _ in range(weight_dim):
            layer_norm.append(nn.LayerNorm(self.dim))
        self.layer_norm = nn.Sequential(*layer_norm)
            
        if no_video:
            embed = 257
        else:
            embed = self.dim
        
        if not crossfeat:
            self.lstm_enc = nn.Sequential(
                nn.Linear(embed, 256, bias=True),
                Blstm(input_size=256, hidden_size=256, num_layers=2),
                nn.Linear(256, 257, bias=True),
                # nn.Sigmoid()
            )
        else:
            self.lstm_enc = nn.Sequential(
                nn.Linear(embed+257, 256, bias=True),
                Blstm(input_size=256, hidden_size=256, num_layers=2),
                nn.Linear(256, 257, bias=True),
                # nn.Sigmoid()
            )            
    
    def forward(self,input_,layer_norm=True):

#             self.weight = self.softmax(self.weight).
        if not self.crossfeat:
            x = input_
            if not self.no_video:
                lms  = torch.split(x, self.dim, dim=2)
                if layer_norm:
                # lmsl = []

                    for i,(lm,layer,weight) in enumerate(zip(lms,self.layer_norm,self.weight)):
                        # pdb.set_trace()
                        if i==0:
                            out = layer(lm)*weight
                        else:
                            out = out+layer(lm)*weight
            else:
                out = x
                    # lmsl.append(layer(lm)*weight)
                # lms = lmsl
            # lms  = torch.cat(lms,-1)
            # pdb.set_trace()
            # x   = (lms*self.softmax(self.weight)).sum(-1)
 
            
            x = self.lstm_enc(out)
        
            return x
        else:
            assert not self.no_video
            x, noisy_spec = input_
            lms  = torch.split(x, self.dim, dim=2)
            if layer_norm:
            # lmsl = []

                for i,(lm,layer,weight) in enumerate(zip(lms,self.layer_norm,self.weight)):
                    # pdb.set_trace()
                    if i==0:

                        out = layer(lm)*weight
                    else:
                        out = out+layer(lm)*weight 
            out = torch.cat((out,noisy_spec),2)
            x = self.lstm_enc(out)

        
            return x



class AVSEHubertModel(torch.nn.Module):
    def __init__(self, ckpt_path, mapping=False, from_scratch=False, no_ssl=False, no_video=False, crossfeat=False):
        super().__init__()
        
        user_dir = "./"
        # utils.import_user_module(Namespace(user_dir=user_dir))
        # if not from_scratch:
        #     models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])

        #     self.model = models[0]
        # else:
        #     _, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        #     self.model = task.build_model(cfg.model)

        # assert not hasattr(self.Model, 'decoder')
        self.mapping = mapping
        self.no_ssl = no_ssl

        if no_ssl:
            import torchvision.models as models
            resnet = models.resnet18(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(512+257, 768*12, bias=True)
        self.no_video = no_video
        self.crossfeat =crossfeat
        self.blstm = BLSTM_WS(no_video,crossfeat)

    def forward(self, video_feats, noisy_feats, noisy_spec):
        if not self.no_ssl:
            # pdb.set_trace()
            feature, layer_results, _ = self.model.extract_finetune(
                source={'video': video_feats, 'audio': noisy_feats}, 
                padding_mask=None, output_layer=12)
            # pdb.set_trace()
            layer_results = [x.transpose(0,1) for (x,z) in layer_results]
            feature       = torch.cat(layer_results,dim=-1)
            B,_,embed_dim = feature.shape

            feature = feature.repeat(1,1,2).reshape(B,-1,embed_dim)

            frame   = noisy_spec.shape[1]
            if frame<=feature.shape[1]:
                feature = feature[:,:frame]
            else:
                feature = torch.cat((feature,feature[:,-1:].repeat(1,frame-feature.shape[1],1)),dim=1)
        else:
            if not self.no_video:

                B, C, T, H, W = video_feats.shape
                #pdb.set_trace()
                assert C==1
                video_feats = video_feats.permute(0,2,1,3,4).view(-1,C,H,W).repeat_interleave(3,dim=1)
                video_feats = self.feature_extractor(video_feats).view(B,T,-1) # B, T, 512
                feature = self.fc(torch.cat((video_feats,noisy_spec),dim=2))
            else:
                feature = noisy_spec


        if not self.mapping:
            if self.crossfeat:
                return self.blstm((feature,noisy_spec))*noisy_spec
            else:
                return self.blstm(feature)*noisy_spec
        else:
            return self.blstm(feature)


        

if __name__ == '__main__':


    model = AVSEHubertModel('C:/Users/batma/Documents/avhubert/base_lrs3_iter5.pt', no_ssl=True)

    video_feats = torch.randn((4, 1, 295, 88, 88))
    noisy_feats = torch.randn((4, 104, 295))
    noisy_spec = torch.randn((4, 295, 257))
    out = model(video_feats, noisy_feats, noisy_spec)
