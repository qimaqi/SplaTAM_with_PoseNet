<!-- PROJECT LOGO -->


## Introduction
Here we copy from the amazing work [Splatam](https://github.com/spla-tam/SplaTAM) and illustrate how to add our proposed PoseNet for better tracking. Please refer to orginal Splatam and our work [Continuous Pose in NeRF](https://github.com/qimaqi/Continuous-Pose-in-NeRF) for more details and usage.

#### Installatation and Data prepare
Here we follow the same as Splatam
SplaTAM has been tested on python 3.10, CUDA>=11.6. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda create -n splatam python=3.10
conda activate splatam
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

Alternatively, we also provide a conda environment.yml file :
```bash
conda env create -f environment.yml
conda activate splatam
```

#### Independences
We add two more scripts to Splatam, you can copy it to your slam-project. ***utils/PoseNet.py*** and ***utils/rotation_conversions.py.***
Then in main scripts ***scripts/splatam.py*** we add 
```python
from utils.PoseNet import TransNet, RotsNet
from utils import rotation_conversions
```


#### Change Tracking
Since now we do not want to initialize independent parameters for tracking, but using time + PoseNet to output pose for optimization, we need to do following modifications:

---
- Step 0:
 - Initialize the PoseNet, we basically just need the number of frames, if you do not know in your case, just put a large number here. If you want to small model, feel free to change the layers_feat
  ```python
    ##### create PoseNet 
    posenet_config = { 'n_img':num_frames , 
                    'tracking': {
                              "poseNet_freq": 5, 
                              "device": "cuda:0", 
                              "layers_feat": [None,256,256,256,256,256,256,256,256],
                              "skip": [4] }
                            }
    transNet = TransNet(posenet_config)
    transNet = transNet.to(device)
    rotsNet = RotsNet(posenet_config)
    rotsNet = rotsNet.to(device)

  ```


- Step 1:
  - Change the optimizer and parameter initialization, we modify the function ***initialize_optimizer()*** to make sure the PoseNet parameters will be passed to optimizer, moreover we set the ``` params['cam_unnorm_rots']``` and ```params['cam_trans'] ``` to ```requires_grad_(False)``` to avoid leaf tensor replacement issues. Note in mapping we change it back and this modificaton might be different based on your implementation in your slam-system.

  ```python
  def initialize_optimizer(params, lrs_dict, tracking, transNet=None, rotsNet=None):
    lrs = lrs_dict
    if tracking:
        if transNet is not None and rotsNet is not None:
            # we do not want to update cam_unnorm_rots=0.0004 and cam_trans=0.002,
            # but we want to update the weights of the networks
            params['cam_unnorm_rots'] = torch.nn.Parameter(params['cam_unnorm_rots'].cuda().float().contiguous().requires_grad_(False))
            params['cam_trans'] = torch.nn.Parameter(params['cam_trans'].cuda().float().contiguous().requires_grad_(False))
            param_groups = [{'params': [v for k, v in params.items() if k not in ["cam_unnorm_rots","cam_trans"]]}]
            param_groups.append({'params': transNet.parameters(), 'name': "transNet", 'lr': lrs['cam_trans']})
            param_groups.append({'params': rotsNet.parameters(), 'name': "rotsNet", 'lr': lrs['cam_unnorm_rots']})
            return torch.optim.Adam(param_groups)
        else:
            return torch.optim.Adam(param_groups)
    else:
        # in mapping we want to set cam_unnorm_rots to be differentiable agai  
        params['cam_unnorm_rots'] = torch.nn.Parameter(params['cam_unnorm_rots'].cuda().float().contiguous().requires_grad_(True))
        params['cam_trans'] = torch.nn.Parameter(params['cam_trans'].cuda().float().contiguous().requires_grad_(True))
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
  ```
---
- Step 2
  - We change the main loop, note that if you want to make the difference every timestep to be continuous, you need to set the ***forwarp_prop*** to False otherwise it already calculate the candidate pose with a const speed assumption.
  - We first save the last time pose:
    ```python
    if time_idx > 0:
      params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
      current_cam_unnorm_rot_est = params['cam_unnorm_rots'][..., time_idx].detach().clone() 
      candidate_cam_tran_est = params['cam_trans'][..., time_idx].detach().clone() 
      current_cam_unnorm_rot_est_mat = rotation_conversions.quaternion_to_matrix(current_cam_unnorm_rot_est)
    ```
    By doing so after each optimization step we will multiple our new estimate pose on this. Note we convert quaternion to matrix but this is not necessary, you can directly do multiplication on quaternion since RotsNet output normalized quaternion

  - Last we calculate new pose estimation and put it back to ```params```
  ```python
  rots_delta_mat = rotation_conversions.quaternion_to_matrix(rots_delta)
  rots_new_est = torch.matmul(current_cam_unnorm_rot_est_mat, rots_delta_mat) 
  params['cam_unnorm_rots'] = params['cam_unnorm_rots'].clone().detach()
  params['cam_trans'] = params['cam_trans'].clone().detach()
  params['cam_unnorm_rots'][..., time_idx] =  rotation_conversions.matrix_to_quaternion(rots_new_est)
  params['cam_trans'][..., time_idx] = (candidate_cam_tran_est + trans_delta)

  ```
  We use ```params['cam_unnorm_rots'] = params['cam_unnorm_rots'].clone().detach()``` to avoid issues of visiting graph twice.

  --- 

That's it, by simply doing so we test on Replica room 0 and results we show on step 500, more results will coming out.

| Method | Rel Pose Error| Pose Error| RMSE | PSNR |  
|-------|----------|----------|----------|----------|
| Splatam | 0.0011 | 0.0089|   0.019  |  32.3667 |
| Splatam+Ours | 0.0002 | 0.0084 | 0.0031 | 37.0875 |


