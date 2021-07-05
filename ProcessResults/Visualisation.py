import numpy as np
import matplotlib.pyplot as plt


def drawLineGraphs(data: dict, caption: str, out_dir: str, x_label: str, y_label: str):
    '''
    Draw a line graphs for data
    :param data: a dictionary maps from a catergory to another X-Y data, X-Y data is also a dictionary
    :param caption: caption of the figure
    :param out_dir: where to save the figures
    :param x_label: label of the x-axis
    :param y_label: label of the y-axis
    :return: None
    '''
    assert len(data.keys()) <= 7
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xlabel(x_label, fontsize='20')
    plt.xticks(fontsize='15')
    plt.ylabel(y_label, fontsize='20')
    plt.yticks(fontsize='15')
    plt.title(caption, fontweight='bold', fontsize='25')

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    markers = ["o", "v", "s", "*", "X", "D", "2"]

    items = data.items()
    for idx, item in enumerate(items):
        cat, xydata = item
        # plt.plot(xydata.keys(), xydata.values(), markers[idx], markersize=12, linestyle='solid', color=colors[idx],
        #          linewidth=4, label=cat)
        plt.plot(xydata.keys(), xydata.values(), linestyle='solid', color=colors[idx],
                 linewidth=4, label=cat)

    plt.legend(fontsize='20')
    plt.savefig(out_dir, bbox_inches='tight')
    plt.close()


# x1 = np.sort(np.random.random(size=10))
# y1 = np.sin(x1)
# x2 = np.sort(np.random.random(size=10))
# y2 = np.cos(x2)
# data = dict()
# # sin data
# sin_map = dict()
# for x, y in zip(x1, y1):
#     sin_map[x] = y
# data['sin'] = sin_map
# # cos data
# cos_map = dict()
# for x, y in zip(x2, y2):
#     cos_map[x] = y
# data['cos'] = cos_map
# drawLineGraphs(data, caption='Test', out_dir='./Test.pdf', x_label='X', y_label='Y')



