# ViCo Challenge Baseline

[[Homepage]](https://project.mhzhou.com/vico)
[[Paper]](https://arxiv.org/abs/2112.13548)
[[Code]](https://github.com/dc3ea9f/vico_challenge_baseline)
[[Full Dataset]](https://1drv.ms/u/s!Ag220j2nXkVsxS0IOdIKNs_ZTOX-?e=3GQ0yG)
[[Challenge]](https://vico-challenge.github.io/)

This repository provides a baseline method for both the [ViCo challenge](https://vico-challenge.github.io/) and [ViCo Project](https://project.mhzhou.com/vico), including vivid talking head video generation and responsive listening head video generation.

Our code is composed of five groups:

- `Deep3DFaceRecon_pytorch`: use for extract 3dmm coefficients. Mainly from [sicxu/Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch), modified following [RenYurui/PIRender](https://github.com/RenYurui/PIRender)
- `preprocess`: scripts for making dataset compatible with our method
- `vico`: our method proposed in paper *Responsive Listening Head Generation: A Benchmark Dataset and Baseline* [arXiv](https://arxiv.org/abs/2112.13548)
- `PIRender`: render 3dmm coefficients to video. Mainly from [RenYurui/PIRender](https://github.com/RenYurui/PIRender) with minor modifications.
- `evaluation`: quantitative analysis for generations, including SSIM, CPBD, PSNR, FID, CSIM, etc.
  - code for CSIM is mainly from [deepinsight/insightface](https://github.com/deepinsight/insightface)
  - code for lip sync evaluation is mainly from [joonson/syncnet_python](https://github.com/joonson/syncnet_python)
  - in [Challenge 2023](https://vico.solutions/challenge/2023), we use [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) to extract landmarks for LipLMD and 3DMM reconstruction.

For end-to-end inference, this [repo](https://github.com/dc3ea9f/face_utils) may be useful.

## For ViCo Project
This repo is created largely for the [challenge](https://vico-challenge.github.io/), while the [full dataset](https://1drv.ms/u/s!Ag220j2nXkVsxS0IOdIKNs_ZTOX-?e=3GQ0yG) released in [ViCo Project](https://project.mhzhou.com/vico) is slightly different from the challenge data. You can use the [script](preprocess/convert.py) to convert:

```bash
python convert.py --anno_file path_to_anno_file --video_folder path_to_videos --audio_folder path_to_audios --target_folder path_to_target_dataset
```

## Train Baseline

### Data Preparation

1. create a workspace

   ```bash
   mkdir vico-workspace
   cd vico-workspace
   ```

2. download dataset from [this link](https://1drv.ms/u/s!Ag220j2nXkVswCXNjZcGk2mGtMnl?e=sArC1M) and unzip `listening_head.zip` to folder `data/`

   ```bash
   unzip listening_head.zip -d data/
   ```

3. reorganize `data/` folder to meet the requirements of [PIRender](https://github.com/RenYurui/PIRender)

   ```bash
   mkdir -p data/listening_head/videos/test
   mv data/listening_head/videos/*.mp4 data/listening_head/videos/test
   ```

4. clone our code

   ```bash
   git clone https://github.com/dc3ea9f/vico_challenge_baseline.git
   ```

5. extract 3d coefficients for video ([[reference]](https://github.com/RenYurui/PIRender/blob/main/DatasetHelper.md))

   1. change directory to `vico_challenge_baseline/Deep3DFaceRecon_pytorch/`

      ```bash
      cd vico_challenge_baseline/Deep3DFaceRecon_pytorch/
      ```

   2. prepare environment following [this](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/73d491102af6731bded9ae6b3cc7466c3b2e9e48#installation)

   3. prepare `BFM/` and `checkpoints/` following [these instructions](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/73d491102af6731bded9ae6b3cc7466c3b2e9e48#prepare-prerequisite-models)

   3. extract facial landmarks from videos

      ```bash
      python extract_kp_videos.py \
        --input_dir ../../data/listening_head/videos/ \
        --output_dir ../../data/listening_head/keypoints/ \
        --device_ids 0,1,2,3 \
        --workers 12
      ```

   4. extract coefficients for videos

      ```bash
      python face_recon_videos.py \
        --input_dir ../../data/listening_head/videos/ \
        --keypoint_dir ../../data/listening_head/keypoints/ \
        --output_dir ../../data/listening_head/recons/ \
        --inference_batch_size 128 \
        --name=face_recon_feat0.2_augment \
        --epoch=20 \
        --model facerecon
      ```

6. extract audios features

   1. change directory to `vico_challenge_baseline/preprocess`

      ```bash
      cd ../preprocess
      ```

   2. install python package `librosa`, `torchaudio` and `soundfile`

   3. extract audio features

      ```bash
      python extract_audio_features.py \
        --input_audio_folder ../../data/listening_head/audios/ \
        --input_recons_folder ../../data/listening_head/recons/ \
        --output_folder ../../data/listening_head/example/features/audio_feats
      ```

7. reorganize video features

   ```bash
   python rearrange_recon_coeffs.py \
     --input_folder ../../data/listening_head/recons/ \
     --output_folder ../../data/listening_head/example/features/video_feats
   ```

8. organize data

   1. compute mean and std for features

      ```bash
      python statistics_mean_std.py ../../data/listening_head/example/features
      ```

   2. organize for training

      ```bash
      mkdir ../../data/listening_head/example/metadata
      cp ../../data/listening_head/train.csv ../../data/listening_head/example/metadata/data.csv
      cd ../vico
      ln -s ../../data/listening_head/example/ ./data
      ```

### Train and Inference

#### Talking Head Generation

1. train baseline

   ```bash
   python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py \
     --batch_size 4 \
     --time_size 90 \
     --max_epochs 500 \
     --lr 0.002 \
     --task speaker \
     --output_path saved/baseline_speaker
   ```

2. inference

   ```bash
   python eval.py \
     --batch_size 4 \
     --output_path saved/baseline_speaker_E500 \
     --resume saved/baseline_speaker/checkpoints/Epoch_500.bin \
     --task speaker
   ```

#### Listening Head Generation

1. train baseline

   ```bash
   python -m torch.distributed.launch --nproc_per_node 4 --master_port 22345 train.py \
     --batch_size 4 \
     --time_size 90 \
     --max_epochs 500 \
     --lr 0.002 \
     --task listener \
     --output_path saved/baseline_listener
   ```

2. inference

   ```bash
   python eval.py \
     --batch_size 4 \
     --output_path saved/baseline_listener_E500 \
     --resume saved/baseline_listener/checkpoints/Epoch_500.bin \
     --task listener
   ```

### Render to Videos

1. change directory to render

   ```bash
   cd ../PIRender
   ```

2. prepare environment for PIRender following [this](https://github.com/RenYurui/PIRender#1-installation)

3. download the trained weights of PIRender following [this](https://github.com/RenYurui/PIRender#inference)

#### Talking Head

1. prepare vox lmdb

   ```bash
   python scripts/prepare_vox_lmdb.py \
     --path ../../data/listening_head/videos/ \
     --coeff_3dmm_path ../vico/saved/baseline_speaker_E500/recon_coeffs/ \
     --out ../vico/saved/baseline_speaker_E500/vox_lmdb/
   ```

2. render to videos

   ```bash
   python -m torch.distributed.launch --nproc_per_node=1 --master_port 32345 inference.py \
     --config ./config/face_demo.yaml \
     --name face \
     --no_resume \
     --input ../vico/saved/baseline_speaker_E500/vox_lmdb/ \
     --output_dir ./vox_result/baseline_speaker_E500
   ```

#### Listening  Head

1. prepare vox lmdb

   ```bash
   python scripts/prepare_vox_lmdb.py \
     --path ../../data/listening_head/videos/ \
     --coeff_3dmm_path ../vico/saved/baseline_listener_E500/recon_coeffs/ \
     --out ../vico/saved/baseline_listener_E500/vox_lmdb/
   ```

2. render to videos

   ```bash
   python -m torch.distributed.launch --nproc_per_node=1 --master_port 42345 inference.py \
     --config ./config/face_demo.yaml \
     --name face \
     --no_resume \
     --input ../vico/saved/baseline_listener_E500/vox_lmdb/ \
     --output_dir ./vox_result/baseline_listener_E500
   ```

### Example Results

[Video Link](https://vico-challenge.github.io/#generations)

## Evaluation

We will evaluate of the quality of generated videos for the following prespectives:

- generation quality (image level): SSIM, CPBD, PSNR
- generation quality (feature level): FID
- identity preserving: Cosine Similarity from Arcface
- expression: expression features L1 distance from 3dmm
- head motion: expression features L1 distance from 3dmm
- lip sync: AV offset and AV confidence from Sync Net
- lip landmark distance

### Generation Quality
#### Evaluate for PSNR and SSIM

```bash
python compute_base_metrics.py --gt_video_folder {} --pd_video_folder {} --anno_file {*.csv} --task {listener,speaker}
```

#### Evaluate for CPBD

```bash
python compute_cpbd.py --pd_video_folder {} --anno_file {*.csv} --task {listener,speaker}
```

#### Evaluate for FID

```bash
python compute_fid.py --gt_video_folder {} --pd_video_folder {} --anno_file {*.csv} --task {listener,speaker}
```

### Identity Preserving

Pretrained model: `ms1mv3_arcface_r100_fp16/backbone.pth` of [this download link](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215577&cid=4A83B6B633B029CC).

```bash
cd arcface_torch/
python compute_csim.py \
  --gt_video_folder {} \
  --pd_video_folder {} \
  --anno_file {} \
  --task {listener,speaker} \
  --weight ms1mv3_arcface_r100_fp16/backbone.pth
```

### Expression & Head Motion

```
mean distance of exp / (angle, trans) 3d cofficients
```

### Lip Sync (speaker only)

```bash
cd lip_sync/
python compute_lipsync.py --pd_video_folder {} --gt_audio_folder {} --anno_file {*.csv}
```

### Landmark Distance of Lips (speaker only)

```bash
python compute_lmd.py --gt_video_folder {} --pd_video_folder {} --anno_file {*.csv}
```

## Citation
If you think this work is helpful for you, please give it a star and citation :)

```bibtex
@InProceedings{zhou2022responsive,
    title={Responsive Listening Head Generation: A Benchmark Dataset and Baseline},
    author={Zhou, Mohan and Bai, Yalong and Zhang, Wei and Yao, Ting and Zhao, Tiejun and Mei, Tao},
    booktitle={Proceedings of the European conference on computer vision (ECCV)},
    year={2022}
}
```
