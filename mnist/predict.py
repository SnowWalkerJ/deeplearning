import pandas as pd
from net import build_net
from prepare.data import load_testset


def reverse_onehot(series):
    return [list(x).index(max(x)) for x in series]


if __name__ == '__main__':
    net = build_net()
    net.load('models/model1')
    test_input = load_testset()
    pred = net.predict(test_input, keep_prob=1.0)
    data = pd.DataFrame({'ImageId': range(1, len(pred)+1), 'Label': reverse_onehot(pred)})
    data.to_csv('output.csv', index=False)
