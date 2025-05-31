import os
import sklearn
from sklearn.datasets import fetch_20newsgroups
from app.recommendation_system import recommendation_system

#set download location of training and testing data
current_directory = os.getcwd()
desired_directory = f'{current_directory}/.venv/sklearn_data'

#loading training data
newsgroups_train = fetch_20newsgroups(subset='train', data_home=desired_directory)
attributes = dir(newsgroups_train)
#['DESCR', 'data', 'filenames', 'target', 'target_names']

#loading testing data
newsgroups_test = fetch_20newsgroups(subset="test", data_home=desired_directory)

#testing number 1
#print(newsgroups_test["data"][0:4])
# print(newsgroups_test["target"][0:4])


def init():
    converter = dict()
    for i in range(20):
        converter[i] = newsgroups_train["target_names"][i]

def index_to_target(number: int):
    return converter[number]


def main():
    init()
    shingle_count = 2
    permutations = 256
    top_k = 5

    rec_sys = recommendation_system(newsgroups_train["data"], newsgroups_train["target"])
    rec_sys.preprocess()
    rec_sys.shingle(shingle_count)
    rec_sys.index(permutations)
    rec_sys.lsh_256()

    test_data = newsgroups_test["data"][0]
    x = rec_sys.query(test_data, top_k)
    print(x)


if __name__ == "__main__":
    main()
