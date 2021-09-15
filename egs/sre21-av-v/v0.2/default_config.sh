#Default configuration parameters for the experiment

face_det_modeldir=InsightFace-PyTorch/retinaface/weights
face_reco_modeldir=exp/face_models/r100-arcface
face_det_model=$face_det_modeldir/mobilenet0.25_Final.pth
face_reco_model=$face_reco_modeldir/insight-face-v3.pt

face_embed_name=r100-v4
face_embed_dir=`pwd`/exp/face_embed/$face_embed_name
