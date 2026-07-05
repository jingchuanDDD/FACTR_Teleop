# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import yaml
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

from utils_data_process import sync_data_slowest, process_image, gaussian_norm, generate_robobuf

@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig):

    input_path = cfg.input_path
    output_path = cfg.output_path
    
    rgb_obs_topics = list(cfg.cameras_topics)
    state_obs_topics = list(cfg.obs_topics)
    action_config = dict(cfg.action_config)
    action_topics = list(action_config.keys())
    
    assert len(state_obs_topics) > 0, "Require low-dim observation topics"
    assert len(rgb_obs_topics) > 0, "Require visual observation topics"
    assert len(action_topics) > 0, "Require visual observation topics"
    
    
    data_folder = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # initialize topics
    all_topics = state_obs_topics + rgb_obs_topics + action_topics
    
    all_episodes = sorted([f for f in data_folder.iterdir() if f.name.startswith('ep_') and f.name.endswith('.pkl')])
    
    trajectories = []
    all_states = []
    all_actions = []
    pbar = tqdm(all_episodes)
    for episode_pkl in pbar:
        with open(episode_pkl, 'rb') as f:
            traj_data = pickle.load(f)
        traj_data, avg_freq = sync_data_slowest(traj_data, all_topics)
        pbar.set_postfix({'avg_freq': f'{avg_freq:.1f} Hz'})

        traj = {}
        num_steps = len(traj_data[action_topics[0]])
        traj['num_steps'] = num_steps
        traj['states'] = np.concatenate([np.array(traj_data[topic]) for topic in state_obs_topics], axis=-1)
        action_list = []
        for topic in action_topics:
            actions = np.array(traj_data[topic])
            action_list.append(actions)
        traj['actions'] = np.concatenate(action_list, axis=-1)
        
        all_states.append(traj['states'])
        all_actions.append(traj["actions"])
        
        for cam_ind, topic in enumerate(rgb_obs_topics):
            enc_images = traj_data[topic]
            processed_images = [process_image(img_enc) for img_enc in enc_images]
            traj[f'enc_cam_{cam_ind}'] = processed_images
        trajectories.append(traj)
        
    # normalize states and actions
    state_norm_stats = gaussian_norm(all_states)
    action_norm_stats = gaussian_norm(all_actions)
    norm_stats = dict(state=state_norm_stats, action=action_norm_stats)
    
    # dump data buffer
    buffer_name = "buf"
    buffer = generate_robobuf(trajectories)
    with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
        pickle.dump(buffer.to_traj_list(), f)
    
    # dump rollout config
    obs_config = {
        'state_topics': state_obs_topics,
        'camera_topics': rgb_obs_topics,
    }
    rollout_config = {
        'obs_config': obs_config,
        'action_config': action_config,
        'norm_stats': norm_stats
    }
    with open(output_dir / "rollout_config.yaml", "w") as f:
        yaml.dump(rollout_config, f, sort_keys=False)
        
if __name__ == "__main__":
    main()