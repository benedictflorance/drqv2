# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np
import mujoco_py

class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, camera='topview'):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None
        self._camera=camera
        self._camera_settings = {
            "lexa": dict(distance=0.6, lookat=[0, 0.65, 0], azimuth=90, elevation=41+180),
            "latco_hammer": dict(distance=0.8, lookat=[0.2, 0.65, -0.1], azimuth=220, elevation=-140),
            "latco_others": dict(distance=2.6, lookat=[1.1, 1.1, -0.1], azimuth=205, elevation=-165)
            }                                            
        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self._env = env
        if self._camera == "lexa" or self._camera == "latco_hammer" or self._camera == "latco_others":
            self._cam = self._camera_settings[self._camera]
        self.record(env)

    def record(self, env):
        if self.enabled:
            if self._camera == "lexa" or self._camera == "latco_hammer" or self._camera == "latco_others":
                self._env.viewer.cam.distance, self._env.viewer.cam.azimuth, self._env.viewer.cam.elevation = self._cam["distance"], self._cam["azimuth"], self._cam["elevation"]
                self._env.viewer.cam.lookat[0], self._env.viewer.cam.lookat[1], self._env.viewer.cam.lookat[2] = self._cam["lookat"][0], self._cam["lookat"][1], self._cam["lookat"][2] 
                self._env.viewer.render(84, 84)
                frame = self._env.viewer.read_pixels(84, 84)[0]
            else:      
                frame = self._env.render(offscreen=True, resolution=(84, 84), camera_name=self._camera)        
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
