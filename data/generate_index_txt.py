import os
import sys


def generate_index_txt(val_path, file_name):
    action_list = os.listdir(val_path)
    action_list.sort()
    print(len(action_list))
    for i in range(len(action_list)):
        video_label_pair = []
        action = action_list[i]
        video_list = os.listdir(os.path.join(val_path, action))
        video_list.sort()
        # print(i)
        for video in video_list:
            video_label_pair.append([os.path.join(action, video), i])

        # print(len(video_label_pair))
        if len(video_label_pair) == 0:
            print(action)
        with open(file_name, 'a+') as f:
            for data in video_label_pair:
                print(data[1])
                f.write(str(data[0]))
                f.write(' ')
                f.write(str(data[1]))
                f.write('\n')


if __name__ == '__main__':
    val_path = "/home/ssd/video_action/kinects400/rawframes_val/"
    file_name = 'kinects400_val.txt'
    generate_index_txt(val_path, file_name)

    val_path = "/home/data/shicaiwei/kinects400/rawframes_train/"
    file_name = 'kinects400_train.txt'
    generate_index_txt(val_path, file_name)
