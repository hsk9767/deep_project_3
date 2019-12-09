import torch
import torch.nn as nn
#12/06
# normalization -1 ~ 1
# edge cv2
# batch size > 1
# kernel size 5 % padding 2 > 3404 / 5000
# fc size > 3265 / 5000
# epoch
# output channel 더 깊게
# layer 상범 버전으로 (stride =2)dropout 제거 > acc : 2946 / 5000
# layer 상범 버전으로 (stride =1)dropout 제거 > 4140 / 5000
# reshape 이상한 4191 / 5000
# 위와 동일 2 epoch 4583 / 5000

#####
##edge 와 single_npy 를 concatenate 해서 channel = 2 로 집어넣는 실험, lr = 0.001 로 높여봄 ->  70퍼센트대로 떨어짐.
##edge 는 버리고 패딩을 없애봄. 어차피 이미지 경계부분이 중요한 것은 아니니까. 또한, 첫 번째 출력의 depth 를 16, 두 번째 출력의 depth 를 32 -> 4136 / 5000 , 5m 15s
##Normalize 를 통해 -1 ~ 1 의 image 로 바꾸고 진행. -> 5m31s , 4315 / 5000
##Normalize 유지해주고 layer1 out 을 12으로 layer2 out 을 24로, weight_decay 추가 ->걸린 시간 : 5m13s, acc : 4114 / 5000
##다시 그럼 16,32 출력을 유지하고 fc layer 의 출력을 100 , 50 으로 바꾼다. -> 7분 15초;;
##그 전에 16, 32 가 반영 안 됐을 수도 있다. 일단 확 낮춰서 CNN : 10, 20  / FC : 100 , 50을 출력으로 해 본다. 0> 7분 13초;;
##Normalize 를 버리고 CNN 출력 10,20, FC : 150,50 으로 -> 근데도 7분 11초..?...
##그러면 다시 fc layer 를 120 으로 cnn 출력 16,32 으로 해봄. -> 근데 또 7분 14초..;;;
##batch :  10, 다시 Normalize 부활 -> 걸린 시간 : 4m6s, acc : 3647 / 5000
##그럼 batch 를 5 로 하고 lr 을 0.005로 높여 보겠음.  걸린 시간 : 4m23s, acc : 3319 / 5000
## batch 1로 하고 lr 을 0.001 로 다시 낮춰 보겠음..걸린 시간 : 7m6s, acc : 4033 / 5000
## batchnorm 삭제 .. 걸린 시간 : 6m14s, acc : 4026 / 5000
## 흠그냥conv layer 3개 넣고 fc layer 3개 넣고 1 epoch .. 걸린 시간 : 3m39s, acc : 2342 / 5000
##cnn 풀력 16, 32 인데 두 번째는 kernel size 3, fc 출력 120 , 50, 2 epoch .. 걸린 시간 : 6m13s, acc : 4625 / 5000, lr = 0.0001
##위에랑 똑같은데 fc out 을 150 / 50 으로 바꿔보기 .. 걸린 시간 : 6m19s , acc : 4602 / 5000
##그럼 fc layer 를 바로 50으로 뽑기 -> bottleneck 현상으로 underfit
##12.08
##CNN (16,16) -> 2번째 cnn 에서만 maxpooling , 두 번째 kernel만 size  3, fc layer 는 (2048, 50) , normalize 반전은 유지, lr = 0.0001, decay = 1e-5 -> 걸린 시간 : 5m27s, acc : 4659 / 5000
##위랑 동일, batch norm 없애기 -> 걸린 시간 : 4m51s , acc : 4752 / 5000
##위랑 동일한데 cnn 출력을 (32, 32) 로 해보기 -> 걸린 시간 : 4m47s, acc : 4815 / 5000
##위랑 동일한데 cnn 출력을 (64, 64) 로 해보기 ->걸린 시간 : 4m45s, acc : 4862 / 5000
## cnn kernel 을 모두 3 size 로. -> 7m33s, acc : 96.6%

##새로운 구조 -> 8분 에 97%
##새로운 구조를 하는데, x_1 은 이미 2048이니까 2048 로 가는 Linear 하지 않고 해 보기. -> 시간이 너무 느렸음.
##새로운 구조를 하는데 batch size 3 -> 걸린 시간 : 4m31s, acc : 0.9678 %
## 50으로 두 개 뽑은 다음에 더할 때에는 -> 걸린 시간 : 4m41s, acc : 0.972 %
##그럼 50으로 뽑은 다음에 더할 때로 해서 fc 단에 leakyrelu 사용 ->

class convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # nn.Conv2d(1, 6, 5, stride = 1, padding = 2),
            nn.Conv2d(1,64,5, stride = 1),
#             nn.BatchNorm2d(16),
            nn.ReLU(),
#             nn.MaxPool2d(2)
            nn.Conv2d(64, 64, 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Linear( 13 * 13 * 64, 2048),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Linear(4*4*128, 50)
        self.layer5 = nn.Linear(2048, 50)
        
# #         self.layer3 = nn.Sequential(
# #             nn.Conv2d(32, 64, 3, stride = 1),
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(),
# #             nn.MaxPool2d(2)
# #         )
#         self.layer4 = nn.Sequential(
            
#         )
# #         self.layer5 = nn.Sequential(
# #             nn.Linear(150, 100),
# #             nn.ReLU()
# #         )
#         self.layer6 = nn.Sequential(
#             # nn.Dropout(0.3),
            
#         )

#     def forward(self, x):
#         x = self.layer1(x)
        
#         x_1 = x.clone()
#         x_1 = self.layer3(x_1)
#         x_1 = x_1.view(-1, 4*4*128)
# #         x_1 = self.layer4(x_1)
        
#         x = x.view(-1, 13 * 13 * 64)
#         x = self.layer2(x)
        
#         x += x_1
        
#         return self.layer5(x)
    def forward(self, x): ##50으로 나온 값을 더하기
        x = self.layer1(x)
        
        x_1 = x.clone()
        x_1 = self.layer3(x_1)
        x_1 = x_1.view(-1, 4*4*128)
        x_1 = self.layer4(x_1)
        
        x = x.view(-1, 13 * 13 * 64)
        x = self.layer2(x)
        x = self.layer5(x)
        
        
        return x + x_1
