import os

from matplotlib import image
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras

import utils


def subsample_and_stack(data, frame_skip=1, stack=4):
    """
    Subsample every n frames, and then stacks every m of these,
    where n = frame_skip and m=stack.
    Subsampling does not seem to be necessary for the stable baselines
    environment, since the Stable Baselines environment
    already includes a frame skip of 4.
    """
    # ToDo: This is memory inefficient since it contains multiple copies of
    #  the same frame.

    assert data.shape[3] == 1
    subsampled_frames = data[::frame_skip, :, :, 0]

    stacked_frames_shape = [
        subsampled_frames.shape[0] - stack + 1,
        subsampled_frames.shape[1],
        subsampled_frames.shape[2],
        stack
    ]
    stacked_frames = np.zeros(shape=stacked_frames_shape)
    for i in range(stacked_frames_shape[0]):
        stacked_frames[i, :, :, :] = np.stack([
            subsampled_frames[i + j] for j in range(stack)
        ], axis=-1)
    return stacked_frames


def load_state_action_pair_traj(image_folder, trajectory_file, frameskip=1,
                                grayscale=True, stack=1, offset=0,
                                ext='png', return_frames=True):
    """
    Same as above, but for generated trajectory data.
    """
    data = np.load(trajectory_file)
    if offset > 0:
        frame_path_list = data['obs'][::frameskip][:-offset]
    else:
        frame_path_list = data['obs'][::frameskip]

    # In new versions of the generated trajectories, there is a column for
    # model selected actions, which we want to predict. This is to distinguish
    # between the chosen actions, which may be a sticky action.
    if 'actions' in data.keys():
        label = 'actions'
    elif 'model selected actions' in data.keys():
        label = 'model selected actions'
    action_list = data[label][:, 0][::frameskip][offset:]
    # Offset = 0 seems to be the correct amount of offset to make actions
    # and frames line up correctly.

    # Need this because I changed the folder structure a bit.
    def _adjust_frame_path(x):
        # ToDo: Right now have this try except code since the format of
        #  the path could be different, if for example the trajectories were
        #  generated locally (in windows) and this is run on google cloud. Would
        #  be good to find a better way to do this.
        try:
            new_path = os.path.join(image_folder, x.split('\\')[-2],
                                    x.split('\\')[-1])
        except:
            new_path = os.path.join(image_folder, x.split('/')[-2],
                                    x.split('/')[-1])
        return new_path

    frame_path_list = [_adjust_frame_path(x) for x in frame_path_list]

    # Remove first stack-1 frames, since there won't be enough frames before
    # those frames to stack.
    frame_path_list = frame_path_list[stack-1:]
    action_list = action_list[stack-1:]

    if return_frames:
        frame_list = load_stacked_frames(frame_path_list, grayscale,
                                         stack=stack, ext=ext)
    else:
        frame_list = None
    return action_list, frame_path_list, frame_list


class DataGenerator(keras.utils.Sequence):
    """
    Generates batches of loaded images. Based off of
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly,
    with some changes.

    Also, instead of maintaining a list of ids, we maintain a list of paths to
    images.
    """

    def __init__(self, path_list, targets, batch_size=32,
                 shuffle=True, ext='png', preprocess_fcn=None,
                 stack=4, grayscale=True):
        """

        :param path_list: list of paths corresponding to frames, which are the
        last frames for each stack.
        :param targets: the labels for each frame stack
        :param batch_size: The size of batches of stacked frames we generate.
        :param shuffle: If set to true, we shuffle the image paths. Note that
        the stacked frames will still be in the correct order, since we only
        shuffle the path of the last frame, and then grab the three proceeding
        frames by subtracting from the frame number.
        :param ext: Extension for the frames (e.g., png, jpg)
        :param preprocess_fcn: An optional function to apply to the images. If
        set to None, no additional preprocessing is done.
        :param stack: Number of frames to stack.
        :param grayscale: Whether the images are read as grayscale or RGB.
        """
        self.path_list = path_list
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert ext in ('jpg', 'png')
        self.ext = ext
        self.preprocess_fcn = preprocess_fcn
        self.stack = stack

        if not grayscale:
            raise NotImplementedError(
                "RGB not implemented yet. Need to combine RGB channels with "
                "framestack channels in order to work with VAE"
            )
        self.grayscale = grayscale

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indices = self.indices[
                  index * self.batch_size:(index + 1) * self.batch_size]

        # # Find list of IDs
        # path_list_temp = [self.path_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(indices)

        return X, y

    def on_epoch_end(self):
        """
        Shuffles the indices after each epoch
        """
        self.indices = np.arange(len(self.path_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, tmp_indices):
        'Generates data containing batch_size samples'
        frame_path_list = [self.path_list[k] for k in tmp_indices]
        target_list = [self.targets[k] for k in tmp_indices]
        frames_list = load_stacked_frames(
            frame_path_list, self.grayscale, self.stack, ext=self.ext
        )

        frames_list = np.array(frames_list)
        target_list = np.array(target_list)

        if self.preprocess_fcn is not None:
            frames_list = self.preprocess_fcn(frames_list)
        return frames_list, target_list


def create_data_loader(main_folder, min_traj, max_traj,
                       num_actions, stack=4, data_ext='npz',
                       image_ext='png', batch_size=32, shuffle=True,
                       preprocess_fcn=None, grayscale=True):
    """
    Code to create a data generator.

    :param main_folder: The main folder to grab trajectories from.
    :param min_traj: Int for min trajectory number.
    :param max_traj: Int for max trajectory number.
    :param num_actions: Number of actions (e.g., number of classification
    classes, used to correctly one hot encode)
    :param stack: Int for number of stacked frames.
    :param data_ext: Extension of data files.
    :param image_ext: Extension of frames.
    :param batch_size:
    :param shuffle:
    :param preprocess_fcn:
    :param grayscale:
    :return: DataLoader object.
    """
    if isinstance(main_folder, list):
        frame_path_list = []
        actions_one_hot = []
        for i in range(len(main_folder)):
            frame_paths, actions = _gather_frame_paths_and_actions(
                main_folder[i], min_traj[i], max_traj[i], num_actions, stack=4,
                data_ext='npz', image_ext='png'
            )
            frame_path_list.extend(frame_paths)
            actions_one_hot.extend(actions)
    else:
        frame_path_list, actions_one_hot = _gather_frame_paths_and_actions(
            main_folder, min_traj, max_traj, num_actions, stack=4,
            data_ext='npz', image_ext='png'
        )

    data_gen = DataGenerator(
        path_list=frame_path_list,
        targets=actions_one_hot,
        batch_size=batch_size,
        shuffle=shuffle,
        ext=image_ext,
        preprocess_fcn=preprocess_fcn,
        stack=stack,
        grayscale=grayscale
    )
    return data_gen


def _gather_frame_paths_and_actions(main_folder, min_traj, max_traj,
                                    num_actions, stack=4, data_ext='npz',
                                    image_ext='png'):
    data_files = [os.path.join(main_folder, f) for f in
                  utils.get_all_files(main_folder)]
    action_list = []
    frame_path_list = []

    def get_trajectory_num(x):
        assert x.split('.')[-1] == data_ext
        return int(x.split('_')[-1][: -(len(data_ext)+1)])

    data_files = [path for path in data_files if
                  min_traj <= get_trajectory_num(path) <= max_traj]
    for file in data_files:
        actions, frame_paths, _ = load_state_action_pair_traj(
            main_folder, file, frameskip=1, grayscale=True, stack=stack,
            offset=0, ext=image_ext, return_frames=False
        )
        action_list.extend(actions)
        frame_path_list.extend(frame_paths)
    actions_one_hot = tf.one_hot(action_list, depth=num_actions)
    return frame_path_list, actions_one_hot


# --------------- Frame Loading Functions ---------------


def load_stacked_frames(frame_path_list, grayscale, stack, ext='png'):
    if stack == 1:
        frame_list = [load_single_image(path, ext=ext, grayscale=grayscale)
                      for path in frame_path_list]
    else:
        frame_list = []

        if not grayscale:
            raise NotImplementedError
        for path in frame_path_list:
            frame_number = _get_frame_number(path)
            directory = _get_folder(path)
            assert frame_number >= stack - 1, \
                print("Frame number {} is too small to stack {} frames. "
                      "Check that the frame path list filters out these "
                      "frames first.".format(frame_number, stack))

            # Wait until we see first image before initializing stacked frames,
            # so that we know the dimensions.
            stacked_frames = None

            for i in range(stack):
                img_path = _get_path_from_frame_number(
                    directory, frame_number - i, ext=ext
                )
                img = load_single_image(img_path, ext=ext, grayscale=grayscale)
                assert img.shape[-1] == 1
                img = img[..., 0]
                if stacked_frames is None:
                    stacked_frames = np.zeros(img.shape+(stack,))
                stacked_frames[..., -i-1] = img
            frame_list.append(stacked_frames)
    return frame_list


def load_single_image(path, ext='png', grayscale=True):
    """
    Loads a single image from a path.
    """
    if not grayscale:
        # Png files read this way will be decimals between 0 and 1.
        img = image.imread(path)
    elif grayscale:
        # Png files read this way will be integers form 0 to 255, so we need
        # to divide by 255 to convert to float between 0 and 1.
        img = Image.open(path).convert("L")
        img = np.array(np.asarray(img))[:, :, np.newaxis]
        img = img / 255.
    if ext == 'jpg':
        raise NotImplementedError("Need to check if dividing by 255 is "
                                  "necessary")
    return img


def _get_frame_number(path):
    """
    Gets the frame number from a path. The frames here are assumed to look like
    'folder\frame_num.ext'
    """
    return int(path.split(os.sep)[-1].split('.')[0])


def _get_folder(path):
    return os.sep.join(path.split(os.sep)[:-1])


def _get_path_from_frame_number(directory, frame_number, ext):
    return os.path.join(directory, '{}.{}'.format(frame_number, ext))
