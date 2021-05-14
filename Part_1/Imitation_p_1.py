import sys
import cv2
import numpy as np
import torch
import os
import math
from MODEL import PoseEstimationWithMobileNet
from KEYPTS import extract_keypoints, group_keypoints
from UTILS import Pose, track_poses, load_state
from UTILS import normalize, pad_width

sys.path.append("./"), sys.path.append("../")


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def write_data(path, d):
    with open(path, 'a') as file:
        file.write(str(d) + "\n")


def files_check(folder_name):
    file_list = ["l_elbow.txt", "l_knee.txt", "l_shoulder.txt", "l_thigh.txt",
                 "r_elbow.txt", "r_knee.txt", "r_shoulder.txt", "r_thigh.txt"]
    path = "../DATA/ANGLES/"
    if os.path.isdir(path + str(folder_name)):
        pass
    else:
        os.mkdir(path + str(folder_name))

    for i in range(len(file_list)):
        if os.path.isfile(path + str(folder_name) + "/" + file_list[i]):
            pass
        else:
            open(path + str(folder_name) + "/" + file_list[i], "x")


def find_angles(pose, mov_name):
    path = "../DATA/ANGLES/" + str(mov_name) + "/"
    files_check(str(mov_name))

    def _slope(p1, p2, st_line):
        if st_line:
            return (p2[1] - (p1[1] + 80)) / (p2[0] - (p1[0] + 5))
        else:
            return (p2[1] - p1[1]) / (p2[0] - p1[0])

    def _angle(m1, m2, degree):
        if degree:
            return round(math.degrees(math.atan((m1 - m2) / (1 + m1 * m2))))
        else:
            math.atan((m1 - m2) / (1 + m1 * m2))

    kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye', 'r_ear', 'l_ear']
    try:
        for i in pose:
            if i.id != 0:
                a = i.keypoints
                # right shoulder
                write_data(path + "r_shoulder.txt", math.radians(_angle(_slope(a[2], a[2], True), _slope(a[2], a[3], False), True)))
                # right elbow
                write_data(path + "r_elbow.txt", math.radians(_angle(_slope(a[2], a[3], False), _slope(a[3], a[4], False), True)))
                # right thigh
                write_data(path + "r_thigh.txt", math.radians(_angle(_slope(a[8], a[8], True), _slope(a[8], a[9], False), True)))
                # right knee
                write_data(path + "r_knee.txt", math.radians(_angle(_slope(a[8], a[9], False), _slope(a[9], a[10], False), True)))
                # left shoulder
                write_data(path + "l_shoulder.txt", math.radians(_angle(_slope(a[5], a[5], True), _slope(a[5], a[6], False), True)))
                # left elbow
                write_data(path + "l_elbow.txt", math.radians(_angle(_slope(a[5], a[6], False), _slope(a[6], a[7], False), True)))
                # left thigh
                write_data(path + "l_thigh.txt", math.radians(_angle(_slope(a[11], a[11], True), _slope(a[11], a[12], False), True)))
                # left knee
                write_data(path + "l_knee.txt", math.radians(_angle(_slope(a[11], a[12], False), _slope(a[12], a[13], False), True)))
    except Exception:
        pass


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

            find_angles(current_poses, "TEST_CASE_4")
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
                #cv2.line(img, (p1, p2), (p1, p2 + 150), (0, 255, 255), 1)
        cv2.imshow('Human_Pose_Estimation', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("./checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)

    frame_provider = VideoReader("../DATA/VIDEOS/walk_female_1.mp4")
    run_demo(net, frame_provider, 256, 0, 1, 1)
