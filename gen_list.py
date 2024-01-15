import random
import os

train_noise = [f'n{noise_type}_{d}db' if d>=0 else f'n{noise_type}_n{-1*d}db' \
                for noise_type in range(1,101) for d in range(-12,13,6) ]

test_types = ['babycry', 'engine', 'music', 'pink', 'street']
test_noise = [f'{noise_type}_n{-1*d}db' if d<0 else f'{noise_type}_{d}db' for noise_type in test_types for d in range(-1,6,3) ]

def select_utter(speakers, noise_types, mode):
    sp = random.choice(speakers)
    noise = random.choice(noise_types)

    if mode == 'train':
        sent_id = random.randrange(190) + 1 
        # print(f"sp: {sp}, sent: {sent_id}, noise: {noise}")

    elif mode == 'val':
        sent_id = random.randrange(190, 200) + 1 
        # print(f"sp: {sp}, sent: {sent_id}, noise: {noise}")
    elif mode == 'test':
        sent_id = random.randrange(200, 320) + 1 
        # print(f"sp: {sp}, sent: {sent_id}, noise: {noise}")

    noisy_utter = os.path.join(sp, 'audio', 'noisy_enh', mode, noise, sp+"_"+f"{sent_id:03d}"+".wav")
    clean_utter = os.path.join(sp, 'audio', 'clean', mode, sp+"_"+f"{sent_id:03d}"+".wav")
    video_utter = os.path.join(sp, 'video', 'mouth', sp+"_"+f"{sent_id:03d}"+".mp4")
    return noisy_utter, clean_utter, video_utter


# def gen_list(num_utter, speakers, noise_types, mode):
    
#     with open(args.output, 'w') as f:
#         f.write(args.data_path + "\n")
#         if mode in ['train', 'val', 'test']:
#             for i in range(num_utter):
#                 noisy_utter, clean_utter, video_utter = select_utter(speakers, noise_types, mode)
#                 f.write(video_utter + "\t" + clean_utter + "\t" + noisy_utter + "\n")



def gen_list(num_utter, speakers, noise_types, mode):
    
    with open(args.output, 'w') as f:
        f.write(args.data_path + "\n")
        if mode in ['train', 'val']:
            for i in range(num_utter):
                noisy_utter, clean_utter, video_utter = select_utter(speakers, noise_types, mode)
                f.write(video_utter + "\t" + clean_utter + "\t" + noisy_utter + "\n")
        elif mode == 'test':
            for sp in args.speaker:
                for noise in noise_types:
                    for sent_id in range(201, 321):
                        noisy_utter = os.path.join(sp, 'audio', 'noisy_enh', mode, noise, sp+"_"+f"{sent_id:03d}"+".wav")
                        clean_utter = os.path.join(sp, 'audio', 'clean', mode, sp+"_"+f"{sent_id:03d}"+".wav")
                        video_utter = os.path.join(sp, 'video', 'mouth', sp+"_"+f"{sent_id:03d}"+".mp4")
                        f.write(video_utter + "\t" + clean_utter + "\t" + noisy_utter + "\n")


def main(args):

    # print(args)

    if args.mode in ['train', 'val']:
        gen_list(args.num_utter, args.speaker, train_noise, args.mode)
    elif args.mode == 'test': 
        gen_list(args.num_utter, args.speaker, test_noise, args.mode)


def parse_args():

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data-path', default='/home/richardleelai/jchou/project/avhubert_avse/corpus/mandarin/')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--output', default='', help='save path')
    parser.add_argument('--speaker', type=str, nargs='+', help='speakers')
    parser.add_argument('--num-utter', default=12000, type=int, help='number of noisy utterances')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
