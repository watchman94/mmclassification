import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('embedding_file', help='embedding_file')
    args = parser.parse_args()

    with open(args.embedding_file, 'r') as openfile:
        embedding_dict = json.load(openfile)

    for k, v in embedding_dict.items():
        embedding_dict[k] = np.array(v)

    correct, error = 0, 0

    out = open('error_list.log', 'w')
    loc_threshold = 5

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
            #pic_name_l = pic_name.split("_")
            if kl[-1] == pic_name_l[-1] and kl[-3] == pic_name_l[-3]:
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

