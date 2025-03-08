import os
# from ctcdecode import CTCBeamDecoder
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np
from wer import calculate_wer
# import torchmetrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from conformer import Conformer
import time
import sentencepiece as spm
from info_nce import InfoNCE, info_nce
from model import *
from data_load import *
import itertools

def multi_step_info_nce(predicted, target, temperature=0.1):
    """
    predicted: [B, future_frames, D]
    target:    [B, future_frames, D]
    We'll flatten to [B*future_frames, D], then do standard InfoNCE across the batch.
    """
    B, T, D = predicted.shape

    # print(target.shape)

    # Flatten
    pred_flat = predicted.reshape(B * T, D)  # [B*T, D]
    targ_flat = target.reshape(B * T, D)     # [B*T, D]

    # L2 normalize
    # pred_flat = F.normalize(pred_flat, p=2, dim=-1)
    # targ_flat = F.normalize(targ_flat, p=2, dim=-1)

    # Similarity
    # logits = torch.matmul(pred_flat, targ_flat.transpose(0, 1))  # => [B*T, B*T]
    # labels = torch.arange(logits.size(0), device=logits.device)
    # logits = logits / temperature

    # loss = F.cross_entropy(logits, labels)
    info_loss = InfoNCE()
    loss = info_loss(pred_flat, targ_flat)

    return loss


class SentencePieceTransform:
    """Maps subwords to integers and vice versa using SentencePiece"""
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def text_to_int(self, text):
        """ Use the SentencePiece tokenizer to convert text to an integer sequence """
        subwords = self.sp.EncodeAsPieces(text.lower())
        return [self.sp.PieceToId(subword) for subword in subwords]

    def int_to_text(self, labels):
        """ Use the SentencePiece tokenizer to convert integer labels to a text sequence """

        return self.sp.decode(labels)

sentencepiece_transform = SentencePieceTransform("spm_unigram.model")

bpe_model = spm.SentencePieceProcessor()
bpe_model.load("spm_unigram.model")
tokens = [bpe_model.id_to_piece(id) for id in range(bpe_model.get_piece_size())]
# print(tokens)


def get_audio_transforms():
  time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=27, p=0.05) for _ in range(10)]
  train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels= 40),  #, hop_length=160
    torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
    *time_masks,
  )

  valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=40)

  return train_audio_transform, valid_audio_transform

train_audio_transforms, valid_audio_transforms = get_audio_transforms()

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    utterances = []

    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(sentencepiece_transform.text_to_int(utterance))
        labels.append(label)
        input_lengths.append(spec.shape[0])
        label_lengths.append(len(label))
        utterances.append(utterance)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths, utterances



def train(model, device, train_loader_c, criterion, enc_optimizer, cpc_optimizer, ctc_optimizer, kwargs, epoch, train_loader2, pre_optimizer, gam):
    
    data_len = len(train_loader2)
    model.train()
    data_len_c = len(train_loader_c.dataset)
 
        
    info_loss = 0
        
    for pre_ep in range(1, 5):

        for batch_idx, data_p in enumerate(train_loader2):
                
                ########################       English #####################################################                       
                        pre_optimizer.zero_grad()
                  
                        contexts, futures = data_p
                        contexts = contexts.to(device)
                        futures = futures.to(device)
                      
                        predicted, target = model(context_wave=contexts, future_wave=futures)
                        cpc_loss = multi_step_info_nce(predicted, target)
                        cpc_loss.backward()
                        pre_optimizer.step()
                    
                        info_loss += loss_cpc.item() / len(train_loader2)
                    
                        if batch_idx % 100 == 0 or batch_idx == data_len:
                            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCPC_Loss: {:.6f}'.format(
                                pre_ep, batch_idx * len(context), data_len,
                                50. * batch_idx / len(train_loader2), loss_cpc.item()))
                # prescheduler.step()

    # for batch_idx, (_data, inputs) in enumerate(zip(train_loader, train_loader2)):

    info_loss = 0.0
    train_loss = 0.0

    for batch_idx, (_data_c,data_p) in enumerate(zip(train_loader_c,train_loader2)):

            gamma = round(gam, 3)
            
            ########################       English #####################################################

            contexts, futures = data_p
            contexts = contexts.to(device)
            futures = futures.to(device)
      
            spectrograms_c, labels_c, input_lengths_c, label_lengths_c = _data_c


            predicted, target = model(context_wave=contexts, future_wave=futures)
            cpc_loss = multi_step_info_nce(predicted, target)
                        
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0)
        
            info_loss += cpc_loss.item() / len(train_loader2)
        
            if batch_idx % 100 == 0 or batch_idx == data_len:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tCPC_Loss: {:.6f}'.format(
                    epoch, batch_idx * len(context), data_len,
                    50. * batch_idx / len(train_loader2), cpc_loss.item()))
                

            spectrograms_cm = spectrograms_c
            labels_cm = labels_c
            input_lengths_cm = input_lengths_c
            label_lengths_cm = label_lengths_c

            
            # print(spectrograms.shape)
            
            spectrograms_cm = torch.squeeze(spectrograms_cm)
                    
                    # print(spectrograms.size())
                    
            spectrograms_cm = spectrograms_cm.transpose(1,2)
                    
                    # print(spectrograms.size())
                    
            labels_cm= torch.LongTensor(labels_cm.long())
                    
            input_lengths_cm=torch.LongTensor(input_lengths_cm)
            label_lengths_cm=torch.LongTensor(label_lengths_cm)
        #             print(label_lengths.type())
            input_lengths_cm = input_lengths_cm.to(device)
            label_lengths_cm = label_lengths_cm.to(device)
            spectrograms_cm, labels_cm = spectrograms_cm.to(device), labels_cm.to(device)

            

            output_cm = model( mel_feat=spectrograms_c, mel_lengths=input_lengths_c) # (batch_size, sequence_length, dim)

                
            input_lengths_c1 = output_cm[1]
                

            output_c = output_cm[0].transpose(0, 1) # (time, batch, n_class)

            ctc_loss = criterion(output_c, labels_cm, input_lengths_c1, label_lengths_cm)

            train_loss += ctc_loss.item() / len(train_loader_c)  

              
            enc_optimizer.zero_grad()
            cpc_optimizer.zero_grad()
            ctc_optimizer.zero_grad()


            enc_params = list(model.encoder.parameters()) + list(model.conv_feature_extractor.parameters())
            enc_grads = torch.autograd.grad(
                outputs=(ctc_loss + gam * cpc_loss),
                inputs=enc_params,
                retain_graph=True  
            )
            
            for p, g in zip(enc_params, enc_grads):
                p.grad = g
            
            enc_optimizer.step()
    
           
            cpc_params = list(model.predictor.parameters())
            cpc_grads = torch.autograd.grad(
                outputs=cpc_loss,
                inputs=cpc_params,
                retain_graph=True 
            )
            for p, g in zip(cpc_params, cpc_grads):
                p.grad = g
            cpc_optimizer.step()
    
           
            ctc_params = list(model.predictor_ctc.parameters())
            ctc_grads = torch.autograd.grad(
                outputs=ctc_loss,
                inputs=ctc_params,
                retain_graph=False 
            )
            for p, g in zip(ctc_params, ctc_grads):
                p.grad = g
            ctc_optimizer.step()
                
            if batch_idx % 100 == 0 or batch_idx == data_len_c:

                print(f"Epoch: {epoch} | Batch: {batch_idx} "
                      f"| CPC Loss: {cpc_loss.item():.4f} "
                      f"| CTC Loss: {ctc_loss.item():.4f} "
                      f"| gamma: {gam:.3f}")

                
    return train_loss



def final_supervised_finetune(
    model,
    device,
    train_loader,
    criterion,
    enc_optimizer,
    ctc_optimizer,
    num_epochs=5,
    lr=1e-5
):
   
    print("Starting final supervised fine-tuning...")

    for g in enc_optimizer.param_groups:
        g['lr'] = lr
    for g in ctc_optimizer.param_groups:
        g['lr'] = lr

    for ep in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, _data_c in enumerate(train_loader):
            spectrograms_c, labels_c, input_lengths_c, label_lengths_c = _data_c
            spectrograms_c = spectrograms_c.to(device)
            labels_c = labels_c.to(device)
            input_lengths_c = input_lengths_c.to(device)
            label_lengths_c = label_lengths_c.to(device)

            enc_optimizer.zero_grad()
            ctc_optimizer.zero_grad()

            output_cm = model(spectrograms_c, input_lengths_c)
            ctc_acts = output_cm[0].transpose(0,1)
            input_lengths_mod = output_cm[1]

            ctc_loss = criterion(ctc_acts, labels_c, input_lengths_mod, label_lengths_c)
            ctc_loss.backward()

            enc_optimizer.step()
            ctc_optimizer.step()

            total_loss += ctc_loss.item()
            if batch_idx % 100 == 0:
                print(f"[FineTune Epoch {ep+1}/{num_epochs} - Batch {batch_idx}] CTC Loss: {ctc_loss.item():.4f}")

        avg = total_loss/len(train_loader_labeled)
        print(f"FineTune epoch {ep+1}: average CTC loss = {avg:.4f}")



def main(learning_rate=5e-4, batch_size=80, epochs=10,
        train_url="train-clean-100", test_url="test-clean"):

    
    hparams = {
        "n_class": 1000,
        "n_feats": 40,
        "stride":2,
        "dropout": 0.05,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }



    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)

    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size= hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    model = Conformer_JUST(
                  input_dim=hparams['n_feats'], 
                  num_encoder_layers=8,
                  num_classes= len(tokens))
   
    model = nn.DataParallel(model)
          
    model.to(device)
#     print(model)
    
    
    
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))


    # opt_backbone = torch.optim.AdamW(self.model.backbone.parameters(), lr=alpha)
    # opt_classifier = torch.optim.AdamW(self.model.classifier_head.parameters(), lr=beta)

          
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
    #                                         steps_per_epoch=int(len(train_loader)),
    #                                         epochs=hparams['epochs'],
    #                                         anneal_strategy='linear')


    
    ####################################Pre training######################################
    
    
    data_dir = "./data/LibriSpeech/train-clean-100"
    
#     data_dir = "/home/ec2-user/SageMaker/data1"

    audio_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:

            if file.lower().endswith('.flac'):  
                audio_files.append(os.path.join(root, file))
  

    print(len(audio_files))                          

    train_dataset2 = RawAudioDataset(
        audio_files=audio_files,
        sample_rate=16000,
        raw_context_samples=32000, 
        raw_future_samples=16000,   
        stride=16000               
    )
      
    train_loader2 = DataLoader(
        dataset=dataset,
        batch_size=64, 
        shuffle=True,
        num_workers=4,
        collate_fn=raw_collate_fn,
        drop_last=True
    )
    
    print(len(train_loader.dataset))
    print(len(train_loader2.dataset))
          
    enc_optimizer = optim.AdamW(model.encoder.parameters(), lr=1e-4)
    cpc_optimizer = optim.AdamW(model.predictor.parameters(), lr=1e-4)
    ctc_optimizer = optim.AdamW(model.predictor_ctc.parameters(), lr=1e-4)
    
    pre_optimizer = optim.AdamW(
        itertools.chain(model.conv_feature_extractor.parameters(), model.encoder.parameters(), model.predictor.parameters()),
        lr=1e-3
    )
    
    
    gamma_max = 1
    gamma_init = 0
    gamma_argmax_step = 500
    if gamma_init > gamma_max:
        gamma_max = gamma_init
        print('Initial gamma is larger than max gamma, proceeding with gamma_max=gamma_init.')
    gam = gamma_init
    step_gam = (gamma_max-gamma_init)/gamma_argmax_step
    
    
    
    train_loss=[]
    test_loss=[]
    Info_loss = []
    cer=[]
    wer=[]
    for epoch in range(1, epochs + 1):
        
#         for i, gpu in enumerate(device_ids):
#             gpu_memory_allocated = torch.cuda.memory_allocated(device)
#             gpu_memory_cached = torch.cuda.memory_cached(device)
#             print(f"GPU {i}: Memory allocated: {gpu_memory_allocated / (1024**3):.2f} GB")
#             print(f"GPU {i}: Memory cached: {gpu_memory_cached / (1024**3):.2f} GB")

        start_time = time.time()
        
        tra_loss = train(
                          model, device,
                          train_loader_c,   
                          criterion,       
                          enc_optimizer,
                          cpc_optimizer,
                          ctc_optimizer,
                          kwargs,
                          epoch,
                          train_loader2,      
                          pre_optimizer,
                          gam
                      )

        
        end_time = time.time()
        training_time = end_time - start_time

        print(f"Training time: {training_time:.2f} seconds")
        
        torch.save(model.state_dict(), 'lib_100_trained.pth')
        gam+= step_gam

        gam = min(gamma_max,gam)
        
        tes_loss, w =  test(model, device, test_loader, criterion, epoch)
        
        train_loss.append(tra_loss)
        test_loss.append(tes_loss)
        wer.append(w)

    final_supervised_finetune(model, device, train_loader_c, criterion, enc_optimizer, ctc_optimizer, num_epochs=5, lr=1e-5)
        
    return train_loss, test_loss, wer

################################################################################################


learning_rate = 5e-4
batch_size = 60
epochs = 200
libri_train_set = "train-clean-100"
libri_test_set = "test-other"

train_loss, test_loss, wer = main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set)



