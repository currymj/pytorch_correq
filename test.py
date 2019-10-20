import torch

routing_correq = torch.tensor([0.00000000e+000,4.49846224e-048,4.49846224e-048,4.49846224e-048
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,4.49846224e-048,1.85185185e-002,1.85185185e-002,1.85185185e-002
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,1.85185185e-002,2.68790836e-230,2.68790836e-230,2.68790836e-230
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000
,2.68790836e-230,0.00000000e+000,0.00000000e+000,0.00000000e+000]).view(4,4,4,4)