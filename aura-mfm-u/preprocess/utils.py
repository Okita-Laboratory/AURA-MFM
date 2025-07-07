import sys
import os
import json
import numpy as np
import pandas as pd
import av
import cv2
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from projectaria_tools.core import data_provider
from projectaria_tools.core import calibration

dataset_root_dir = '/data4/dataSpace/EGO_EXO4D/'


#################
#      IMU      #
#################
''' muteLog()
projectaria_tools は C++ レベルでログが制御されており、stderrに出力されるようである。(python側でのログ制御が容易ではない)
何も対策をしなければ、projectaria_toolsに関するログがデータを読み込む(providerの操作)ごとに出力される。
`python sample.py 2> /dev/null`このコマンドを叩けば標準エラー出力が廃棄される。
以下のmuteLogクラスは`python sample.py`だけで同じことを実現するために定義したクラスである。'''
class muteLog:
    def __init__(self):
        self.devnull = os.open(os.devnull, os.O_WRONLY) # /dev/null を開く
        self.stderr_fd = sys.stderr.fileno()            # 元の標準エラー (stderr) のファイルディスクリプタ
        self.saved_stderr_fd = os.dup(self.stderr_fd)   # を保存
    def redirect(self):
        os.dup2(self.devnull, self.stderr_fd)           # 標準エラー (fd=2) を /dev/null にリダイレクト
    def close(self):
        os.dup2(self.saved_stderr_fd, self.stderr_fd)   # 保存しておいたファイルディスクリプタを復元
        os.close(self.saved_stderr_fd)                  # 不要になったファイルディスクリプタを閉じる
        os.close(self.devnull)                          # /dev/null を閉じる

''' get_imu()
takeを受け取りそのtake_nameに対応するvrsファイルからimu(acc, gyro)データを取り出す。'''
def get_imu(take, imu_part='imu-left', hz=100):
    # mute = muteLog()
    # mute.redirect()

    step = 1000//hz if imu_part=='imu-right' else 800//hz # 計測器の周波数が1Khz
    ego_exo_project_path = os.path.join(dataset_root_dir, 'takes', take['take_name'])
    vrs_file_path = os.path.join(ego_exo_project_path, 'aria01_noimagestreams.vrs')
    if not os.path.exists(vrs_file_path):
        return False
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file_path)
    stream_id = vrs_data_provider.get_stream_id_from_label(imu_part)
    num_data = vrs_data_provider.get_num_data(stream_id)

    timestamps, imus = [],[]
    for index in range(0, num_data, step): # 1kHzらしい(私の見る限り800Hz)のでstepを設定しサンプリングレートを可変にする。
        imu = vrs_data_provider.get_imu_data_by_index(stream_id, index)
        imus.extend([imu.accel_msec2 + imu.gyro_radsec])
        timestamps.append(imu.capture_timestamp_ns)
    
    # mute.close()
    return np.array(timestamps), np.array(imus)


#################
#     MoCap     #
#################
KEYPOINTS = ['nose', 'left-eye', 'right-eye', 'left-ear', 'right-ear', 'left-shoulder', 'right-shoulder', 'left-elbow', 'right-elbow', 'left-wrist', 'right-wrist', 'left-hip', 'right-hip', 'left-knee', 'right-knee', 'left-ankle', 'right-ankle']
palette = np.array([[255, 128, 0],[255, 153, 51],[255, 178, 102],[230, 230, 0],[255, 153, 255],[153, 204, 255],[255, 102, 255],[255, 51, 255],[102, 178, 255],[51, 153, 255],[255, 153, 153],[255, 102, 102],[255, 51, 51],[153, 255, 153],[102, 255, 102],[51, 255, 51],[0, 255, 0],[0, 0, 255],[255, 0, 0],[255, 255, 255],])
def load_csv_to_df(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as csv_file:
        return pd.read_csv(csv_file)

def get_coords(annot):
    pts = dict()
    for k in annot:
        atype = 1
        if 'palcement' in annot[k].keys() and annot[k]["placement"] == "auto":
            atype = 0
        pts[k] = [annot[k]["x"], annot[k]["y"], atype]
    return pts

def get_body_metadata():
    keypoints_map = [
        {"label": "Nose", "id": "fee3cbd2", "color": "#f77189"},
        {"label": "Left-eye", "id": "ab12de34", "color": "#d58c32"},
        {"label": "Right-eye", "id": "7f2g1h6k", "color": "#a4a031"},
        {"label": "Left-ear", "id": "mn0pqrst", "color": "#50b131"},
        {"label": "Right-ear", "id": "yz89wx76", "color": "#34ae91"},
        {"label": "Left-shoulder", "id": "5a4b3c2d", "color": "#37abb5"},
        {"label": "Right-shoulder", "id": "e1f2g3h4", "color": "#3ba3ec"},
        {"label": "Left-elbow", "id": "6i7j8k9l", "color": "#bb83f4"},
        {"label": "Right-elbow", "id": "uv0wxy12", "color": "#f564d4"},
        {"label": "Left-wrist", "id": "3z4ab5cd", "color": "#2fd4aa"},
        {"label": "Right-wrist", "id": "efgh6789", "color": "#94d14f"},
        {"label": "Left-hip", "id": "ijklmnop", "color": "#b3d32c"},
        {"label": "Right-hip", "id": "qrstuvwx", "color": "#f9b530"},
        {"label": "Left-knee", "id": "yz012345", "color": "#83f483"},
        {"label": "Right-knee", "id": "6bc7defg", "color": "#32d58c"},
        {"label": "Left-ankle", "id": "hijk8lmn", "color": "#3ba3ec"},
        {"label": "Right-ankle", "id": "opqrs1tu", "color": "#f564d4"},
    ]

    # pyre-ignore
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    skeleton = [
                [16, 14],
                [14, 12],
                [17, 15],
                [15, 13],
                [12, 13],
                [6, 12],
                [7, 13],
                [6, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [9, 11],
                [2, 3],
                [1, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 7],
            ]
    return keypoints_map, skeleton, pose_kpt_color

def draw_skeleton(img, all_pts, skeleton):
    draw = ImageDraw.Draw(img)
    for item in skeleton:
        left_index = item[0] - 1
        right_index = item[1] - 1
        left_pt = all_pts[left_index]
        right_pt = all_pts[right_index]
        if len(left_pt) == 0 or len(right_pt) == 0:
            continue
        draw.line([left_pt, right_pt], fill="white", width=10)

def draw_cross(img, x, y, color):
    draw = ImageDraw.Draw(img)
    # Circle parameters
    center = (x, y)  # Center of the cross
    cross_length = 10  # Half-length of the cross arms
    # Calculate the end points of the cross
    left_point = (center[0] - cross_length, center[1])
    right_point = (center[0] + cross_length, center[1])
    top_point = (center[0], center[1] - cross_length)
    bottom_point = (center[0], center[1] + cross_length)

    # Draw the horizontal line
    draw.line([left_point, right_point], fill=color, width=3)
    # Draw the vertical line
    draw.line([top_point, bottom_point], fill=color, width=3)

def draw_circle(img, x, y, color):
    draw = ImageDraw.Draw(img)
    # Circle parameters
    center = (x, y)  # Center of the circle
    radius = 12  # Radius of the circle

    # Calculate the bounding box of the circle
    left_up_point = (center[0] - radius, center[1] - radius)
    right_down_point = (center[0] + radius, center[1] + radius)

    # Draw the circle with a black outline
    draw.ellipse(
        [left_up_point, right_down_point], outline=(255, 255, 255), fill=color, width=6
    )

def get_frame(video_local_path, frame_idx):
    container = av.open(video_local_path)
    print(container.streams.video[0].frames)
    frame_count = 0
    for frame in tqdm(container.decode(video=0)):
        if frame_count == frame_idx:
            input_img = np.array(frame.to_image())            
            pil_img = Image.fromarray(input_img)
            print(frame_count)
            return pil_img
        frame_count+=1

def undistort_exocam(image, intrinsics, distortion_coeffs, dimension=(3840, 2160)):
    DIM = dimension
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    dim1 = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort

    # Change the calibration dim dynamically (bouldering cam01 and cam04 are verticall for examples)
    if DIM[0] != dim1[0]:
        DIM = (DIM[1], DIM[0])

    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = (
        intrinsics * dim1[0] / DIM[0]
    )  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, distortion_coeffs, dim2, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2
    )
    undistorted_image = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return undistorted_image, new_K

def get_distortion_and_intrinsics(_raw_camera):
    intrinsics = np.array(
        [
            [_raw_camera["intrinsics_0"], 0, _raw_camera["intrinsics_2"]],
            [0, _raw_camera["intrinsics_1"], _raw_camera["intrinsics_3"]],
            [0, 0, 1],
        ]
    )
    distortion_coeffs = np.array(
        [
            _raw_camera["intrinsics_4"],
            _raw_camera["intrinsics_5"],
            _raw_camera["intrinsics_6"],
            _raw_camera["intrinsics_7"],
        ]
    )
    return distortion_coeffs, intrinsics

def get_frame_img(take, frame_num, cam):
    exo_traj_path = os.path.join(dataset_root_dir, take['root_dir'], 'trajectory', 'gopro_calibs.csv')
    exo_traj_df = load_csv_to_df(exo_traj_path)
    calib_df = exo_traj_df[exo_traj_df.cam_uid == cam].iloc[0].to_dict()
    D, I = get_distortion_and_intrinsics(calib_df)

    local_path = os.path.join(dataset_root_dir, take['root_dir'], take['frame_aligned_videos'][cam]['0']['relative_path'])
    video = get_frame(local_path, int(frame_num))
    undistorted_frame, _ = undistort_exocam(np.array(video), I, D)
    return undistorted_frame

def get_viz(img, ann):
    keypoints_map, skeleton, pose_kpt_color = get_body_metadata()

    pts = get_coords(ann)
    all_pts = []
    for index, keypoints in enumerate(keypoints_map):
        kpname = keypoints["label"].lower()
        pt = (pts[kpname][0], pts[kpname][1]) if kpname in pts else ()
        all_pts.append(pt)
    
    draw_skeleton(img, all_pts, skeleton)

    for index, keypoints in enumerate(keypoints_map):
        kpname = keypoints["label"].lower()
        if kpname in pts:
            x, y, pt_type = pts[kpname][0], pts[kpname][1], pts[kpname][2]
            color = tuple(pose_kpt_color[index])
            if pt_type == 1:
                draw_circle(img, x, y, color)
            else:
                draw_cross(img, x, y, color)
        else:
            pass
    return img

def get_mocap_npdata(annotation, dim='3D', cam=None):
    d = 2 if dim == '2D' else 3 if dim == '3D' else 0
    ann = (
        annotation[0]['annotation2D'][cam] if dim == '2D' else
        annotation[0]['annotation3D'] if dim == '3D' else None
    )
    keypoints_map, _, _ = get_body_metadata()
    
    pts = get_coords(ann)
    all_pts = []
    for index, keypoints in enumerate(keypoints_map):
        kpname = keypoints["label"].lower()
        pt = [pts[kpname][i] for i in range(d)] if kpname in pts else [np.nan for _ in range(d)]
        all_pts.extend(pt)
    return all_pts

def get_mocap(take, dim='3D', train_or_val='train', cam=None, with_video=False, return_frames=False):
    cam = take['best_exo'] if cam is None else cam
    tuid = take['take_uid']

    take_json_path = os.path.join(dataset_root_dir, f"annotations/ego_pose/{train_or_val}/body/annotation/{tuid}.json")
    if not os.path.exists(take_json_path):
        # print('Error: not exist.')
        return False
    annotations = json.load(open(take_json_path))

    if return_frames:
        frames = {}
        for frame_num, annotation in tqdm(annotations.items()):
            ann = annotation[0]['annotation2D'][cam]
            img = get_frame_img(take, frame_num, cam) if with_video else np.zeros((2160, 3840, 3), dtype=np.uint8)
            frame = get_viz(Image.fromarray(img), ann)
            frames[frame_num] = frame
        return frames
    
    mocaps = []
    for annotation in annotations.values():
        mocaps.append(get_mocap_npdata(annotation, dim=dim, cam=cam))
    mocaps = pd.DataFrame(mocaps).interpolate(limit_direction='both').values.transpose(1, 0)

    frame_nums = np.array(list((map(int, annotations.keys()))))
    return frame_nums, mocaps

def save_gif(imgs, filename='out.gif', duration=80):
    imgs[0].save(filename, save_all=True, append_images=imgs[1:], optimize=False, duration=duration, loop=0)







