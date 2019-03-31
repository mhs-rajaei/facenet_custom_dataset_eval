#! encoding: utf-8

import os
import random

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.

    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """
    counter = 1

    def __init__(self, data_dir, pairs_filepath, img_ext, num_random_images_per_folder):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext
        self.num_random_images_per_folder = num_random_images_per_folder

        if os.name == 'nt':
            self.separator = "\\"
        else:
            self.separator = "/"

        self.remaining = []
        for name in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, name)):
                self.remaining.append(name)

    def update_remaining(self):
        self.remaining = []
        for name in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, name)):
                self.remaining.append(name)

    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()


    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        # for name in os.listdir(self.data_dir):
        for name in self.remaining:
            # if name == ".DS_Store":
            #     continue
            # if os.path.isdir(os.path.join(self.data_dir, name)):
            a = []
            for file in os.listdir(os.path.join(self.data_dir, name)):
            # for file in os.listdir(self.data_dir):
            #     if file == ".DS_Store":
            #         continue
                if self.img_ext in file:
                    # a.append(file)
                    a.append(os.path.join(name, file))

            if a:
                with open(self.pairs_filepath, "a") as f:
                    for i in range(self.num_random_images_per_folder):
                        temp = random.choice(a).split(self.separator)  # This line may vary depending on how your images are named.
                        w = self.separator.join(temp[:-1])

                        # l = random.choice(a).split(self.separator)[-1].lstrip("0").rstrip(self.img_ext)
                        l = random.choice(a).split(self.separator)[-1]
                        # r = random.choice(a).split(self.separator)[-1].lstrip("0").rstrip(self.img_ext)
                        r = random.choice(a).split(self.separator)[-1]

                        print("For '" + os.path.join(self.data_dir, name) + "' and counter: ", self.counter, ', Match Pair:', w + " -> " + l
                              + ", " + r)

                        f.write(w + "\t" + l + "\t" + r + "\n")
                        self.counter += 1
                        # w = temp[0] + "_" + temp[1]
                        # w = '_'.join(temp[:-1])
                        # try:
                        #     l = random.choice(a).split("_")[-1].lstrip("0").rstrip(self.img_ext)
                        #     r = random.choice(a).split("_")[-1].lstrip("0").rstrip(self.img_ext)
                        #
                        #     print(w + " " + l + " " + r, 'counter: ', self.counter)
                        #
                        #     if 'William_Ford' in a:
                        #         print()
                        #
                        #
                        #     f.write(w + "\t" + l + "\t" + r + "\n")
                        #     self.counter += 1
                        # except Exception as err:
                        #     print(err)
                        #     # print('Setting index to 1')
                        #     # l = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                        #     # l = random.choice(a).split.lstrip("0").rstrip(self.img_ext)
                        #     # r = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                        #     # f.write(w + "\t" + l + "\t" + r + "\n")

    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        # for i, name in enumerate(os.listdir(self.data_dir)):
        for i, name in enumerate(self.remaining):
                # remaining = os.listdir(self.data_dir)
                self.update_remaining()
                # remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]
                del self.remaining[i]  # deletes the file from the list, so that it is not chosen again
                other_dir = random.choice(self.remaining)
                with open(self.pairs_filepath, "a") as f:
                    for _ in range(self.num_random_images_per_folder):

                        temps_file_1 = os.listdir(os.path.join(self.data_dir, name))
                        if temps_file_1:
                            file1 = random.choice(temps_file_1)

                        temps_file_2 = os.listdir(os.path.join(self.data_dir, other_dir))
                        if temps_file_2:
                            file2 = random.choice(temps_file_2)

                        if file1 and file2:
                            if self.img_ext in file1 and self.img_ext in file2:
                                print("For '" + self.data_dir + "' and counter: ", self.counter, ', MisMatch Pair:',
                                      name + " " + file1.split(self.separator)[-1] + ' ' +
                                      other_dir + ' ' + file2.split(self.separator)[-1])

                                f.write(name + "\t" + file1.split(self.separator)[-1] + "\t" + other_dir + "\t" +
                                        file2.split(self.separator)[-1] + "\n")

                                self.counter += 1


if __name__ == '__main__':
    # data_dir = "my_own_datasets/"
    # data_dir = r'F:\Documents\JetBrains\PyCharm\OFR\images\200END_lfw_160_test_Copy'
    data_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\All VIS_160"
    pairs_filepath = r"F:\Documents\JetBrains\PyCharm\OFR\original_facenet\data\All_VIS_160_pairs.txt"
    # img_ext = ".jpg"
    img_ext = ".png"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext, num_random_images_per_folder=10)
    generatePairs.generate()

