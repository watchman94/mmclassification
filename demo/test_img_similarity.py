import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('embedding_file', help='embedding_file')
    parser.add_argument('--loc_threshold', type=int, default=5, help='loc_threshold')
    parser.add_argument('--loc_threshold1', type=int, default=2, help='loc_threshold1')
    args = parser.parse_args()

    with open(args.embedding_file, 'r') as openfile:
        embedding_dict = json.load(openfile)

    for k, v in embedding_dict.items():
        embedding_dict[k] = np.array(v)

    correct, error = 0, 0

    out = open('error_list.log', 'w')
    loc_threshold = args.loc_threshold

    for file in list(embedding_dict.keys()):
        pic_name = os.fsdecode(file)
        pic_name_l = pic_name.split("_")

        pic_loc = int(pic_name_l[0][:-1])
        pic_dir = pic_name_l[2].split('.')[0]

        sims = {}
        for key in list(embedding_dict.keys()):
            if key == pic_name:  continue

            #use location and direction to filter
            kl = key.split('_')
            kl_loc = int(kl[0][:-1])
            kl_dir = kl[2].split('.')[0]
            #pdb.set_trace()
            if kl_dir != pic_dir: continue
            if abs(kl_loc - pic_loc) > loc_threshold: continue

            sims[key] = cosine_similarity(embedding_dict[pic_name].reshape((1, -1)), embedding_dict[key].reshape((1, -1)))[0][0]
        d_view = [(v, k) for k, v in sims.items()]
        d_view.sort(reverse=True)
        for v, k in d_view:
            kl = k.split("_")
            loc1 = int(kl[-3][:-1])
            loc2 = int(pic_name_l[-3][:-1])
            if abs(loc1 - loc2) <= args.loc_threshold1 and kl[-1] == pic_name_l[-1]:
                correct += 1
            else:
                error += 1
                line = pic_name + " " + k + " " + str(v) + "\n"
                out.write(line)
            break

    print("correct: " , correct)
    print("error: " , error)
    print("ac: " , (correct * 1.0) / (correct + error))
    out.close()

if __name__ == '__main__':
    main()

