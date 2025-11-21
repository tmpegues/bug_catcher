"""
SORT: A Simple, Online and Realtime Tracker.

Copyright (C) 2016-2020 Alex Bewley <alex@bewley.ai>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import argparse
import glob
import os
import time

import cv2

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import linear_sum_assignment

from skimage import io

matplotlib.use('TkAgg')

np.random.seed(0)


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Solve linear assignment for the given cost matrix."""
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute IOU between two sets of bboxes in the form [x1, y1, x2, y2].

    Args
    ----
    bb_test : np.ndarray
        Bounding boxes to test (N x 4).
    bb_gt : np.ndarray
        Ground truth bounding boxes (M x 4).

    Returns
    -------
    np.ndarray
        IOU matrix of shape (N, M).

    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    denom = (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )

    return wh / denom


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert bbox from [x1, y1, x2, y2] to [x, y, s, r].

    Args
    ----
    bbox : np.ndarray
        Bounding box [x1, y1, x2, y2].

    Returns
    -------
    np.ndarray
        Column vector [x, y, s, r]^T where:
        x, y : box center
        s    : scale (area)
        r    : aspect ratio (w / h)

    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + (w / 2.0)
    y = bbox[1] + (h / 2.0)
    s = w * h  # scale is area
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """
    Convert state vector [x, y, s, r] to bbox [x1, y1, x2, y2].

    Args
    ----
    x : np.ndarray
        State vector [x, y, s, r]^T (4x1 or length-4 array).
    score : float, optional
        Detection score to append.

    Returns
    -------
    np.ndarray
        Bbox [x1, y1, x2, y2] or [x1, y1, x2, y2, score].

    """
    w = np.sqrt(max(x[2] * x[3], 1e-6))
    h = x[2] / w
    x1 = x[0] - (w / 2.0)
    y1 = x[1] - (h / 2.0)
    x2 = x[0] + (w / 2.0)
    y2 = x[1] + (h / 2.0)

    if score is None:
        return np.array(
            [x1, y1, x2, y2],
            dtype=np.float32,
        ).reshape((1, 4))

    return np.array(
        [x1, y1, x2, y2, score],
        dtype=np.float32,
    ).reshape((1, 5))


# #################### Begin_Citation [12] ##################
class KalmanBoxTracker:
    """
    Internal state of an individual tracked object (single bbox).

    Uses an OpenCV KalmanFilter with a constant-velocity model in the
    [x, y, s, r] space, where s is area and r is aspect ratio.
    """

    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        """
        Initialise a tracker using an initial bounding box.

        Args:
        ----
        bbox : np.ndarray
            Bounding box [x1, y1, x2, y2].

        """
        # 7D state: [x, y, s, r, vx, vy, vs]
        # 4D measurement: [x, y, s, r]
        self.kf = cv2.KalmanFilter(7, 4)

        # State transition matrix F
        self.kf.transitionMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Measurement matrix H
        self.kf.measurementMatrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        # Process noise covariance Q
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32)
        self.kf.processNoiseCov[-1, -1] *= 0.01
        self.kf.processNoiseCov[4:, 4:] *= 0.01

        # Measurement noise covariance R
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32)
        self.kf.measurementNoiseCov[2:, 2:] *= 10.0

        # Posterior error covariance P
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)
        self.kf.errorCovPost[4:, 4:] *= 1000.0
        self.kf.errorCovPost *= 10.0

        # Initial state x from bbox
        initial_state = convert_bbox_to_z(bbox).reshape(
            4,
        )
        self.kf.statePost[:4, 0] = initial_state

        self.time_since_update = 0
        self.history: list[np.ndarray] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def update(self, bbox: np.ndarray) -> None:
        """
        Update the state vector with an observed bounding box.

        Args:
        ----
        bbox : np.ndarray
            Observed bounding box [x1, y1, x2, y2].

        """
        self.time_since_update = 0
        self.history.clear()
        self.hits += 1
        self.hit_streak += 1

        z = convert_bbox_to_z(bbox).astype(np.float32)
        self.kf.correct(z)

    def predict(self) -> np.ndarray:
        """
        Advance the state vector and return the predicted bbox.

        Returns
        -------
        np.ndarray
            Predicted bbox [x1, y1, x2, y2] as a (1, 4) array.

        """
        # Prevent scale from becoming negative
        if (self.kf.statePost[6] + self.kf.statePost[2]) <= 0:
            self.kf.statePost[6] = 0.0

        pred = self.kf.predict()

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        bbox = convert_x_to_bbox(pred)
        self.history.append(bbox)
        return bbox

    def get_state(self) -> np.ndarray:
        """
        Return the current bounding box estimate.

        Returns
        -------
        np.ndarray
            Current bbox [x1, y1, x2, y2] as a (1, 4) array.

        """
        return convert_x_to_bbox(self.kf.statePost)

    # #################### End_Citation [12] ##################


def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign detections to tracked objects (both as bboxes).

    Args
    ----
    detections : np.ndarray
        Detections [[x1, y1, x2, y2, score], ...].
    trackers : np.ndarray
        Predicted tracker boxes [[x1, y1, x2, y2, 0], ...].
    iou_threshold : float, optional
        Minimum IOU to accept a match.

    Returns
    -------
    matches : np.ndarray
        Array of matched indices (N x 2): [detection_idx, tracker_idx].
    unmatched_detections : np.ndarray
        Indices of detections with no match.
    unmatched_trackers : np.ndarray
        Indices of trackers with no match.

    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections: list[int] = []
    for d, _ in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers: list[int] = []
    for t, _ in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IOU
    matches: list[np.ndarray] = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches_arr = np.empty((0, 2), dtype=int)
    else:
        matches_arr = np.concatenate(matches, axis=0)

    return (
        matches_arr,
        np.array(unmatched_detections, dtype=int),
        np.array(unmatched_trackers, dtype=int),
    )


class Sort:
    """SORT tracker: Simple Online and Realtime Tracking."""

    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        """
        Set key parameters for SORT.

        Args:
        ----
        max_age : int
            Maximum number of frames to keep a track alive without
            associated detections.
        min_hits : int
            Minimum number of associated detections before a track is
            initialised.
        iou_threshold : float
            Minimum IOU for a valid match.

        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray | None = None) -> np.ndarray:
        """
        Update tracker with newly detected bounding boxes.

        Args
        ----
        dets : np.ndarray, optional
            Detections [[x1, y1, x2, y2, score], ...]. Use
            np.empty((0, 5)) when there are no detections.

        Returns
        -------
        np.ndarray
            Array [[x1, y1, x2, y2, id], ...] of active tracks.

        """
        if dets is None:
            dets = np.empty((0, 5))

        self.frame_count += 1

        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5), dtype=np.float32)
        to_del: list[int] = []
        ret: list[np.ndarray] = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0.0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        (
            matched,
            unmatched_dets,
            unmatched_trks,
        ) = associate_detections_to_trackers(
            dets,
            trks,
            self.iou_threshold,
        )

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialise new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(dets[det_idx, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive IDs
                track_row = np.concatenate(
                    (d, [trk.id + 1]),
                ).reshape(1, -1)
                ret.append(track_row)

            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 5), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """Parse input arguments for the standalone SORT demo."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument(
        '--display',
        dest='display',
        help='Display online tracker output (slow) [False]',
        action='store_true',
    )
    parser.add_argument(
        '--seq_path',
        help='Path to detections.',
        type=str,
        default='data',
    )
    parser.add_argument(
        '--phase',
        help='Subdirectory in seq_path.',
        type=str,
        default='train',
    )
    parser.add_argument(
        '--max_age',
        help=('Maximum number of frames to keep alive a track without associated detections.'),
        type=int,
        default=1,
    )
    parser.add_argument(
        '--min_hits',
        help=('Minimum number of associated detections before track is initialised.'),
        type=int,
        default=3,
    )
    parser.add_argument(
        '--iou_threshold',
        help='Minimum IOU for match.',
        type=float,
        default=0.3,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    display = args.display
    phase = args.phase

    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)

    if display:
        if not os.path.exists('mot_benchmark'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n'
                '    Create a symbolic link to the MOT benchmark\n'
                '    (https://motchallenge.net/data/2D_MOT_2015/#download). '
                'E.g.:\n\n'
                '    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 '
                'mot_benchmark\n\n',
            )
            raise SystemExit(1)

        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')

    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
        )

        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*') :].split(os.path.sep)[0]     # noqa: E203

        out_path = os.path.join('output', f'{seq}.txt')
        with open(out_path, 'w', encoding='utf-8') as out_file:
            print(f'Processing {seq}.')
            max_frame = int(seq_dets[:, 0].max())

            for frame in range(max_frame):
                frame += 1  # Detection and frame numbers begin at 1
                mask = seq_dets[:, 0] == frame
                dets = seq_dets[mask, 2:7]

                # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if display:
                    img_path = os.path.join(
                        'mot_benchmark',
                        phase,
                        seq,
                        'img1',
                        f'{frame:06d}.jpg',
                    )
                    im = io.imread(img_path)
                    ax1.imshow(im)
                    plt.title(f'{seq} Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(
                        '{:d},{:d},{:.2f},{:.2f},{:.2f},{:.2f},1,-1,-1,-1'.format(
                            frame,
                            int(d[4]),
                            d[0],
                            d[1],
                            d[2] - d[0],
                            d[3] - d[1],
                        ),
                        file=out_file,
                    )
                    if display:
                        d_int = d.astype(np.int32)
                        ax1.add_patch(
                            patches.Rectangle(
                                (d_int[0], d_int[1]),
                                d_int[2] - d_int[0],
                                d_int[3] - d_int[1],
                                fill=False,
                                lw=3,
                                ec=colours[int(d[4]) % 32, :],
                            ),
                        )

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    if total_time > 0.0:
        fps = total_frames / total_time
    else:
        fps = 0.0

    print(
        'Total Tracking took: {:.3f} seconds for {} frames or {:.1f} FPS'.format(
            total_time, total_frames, fps
        ),
    )

    if display:
        print(
            'Note: to get real runtime results run without the option: --display',
        )
