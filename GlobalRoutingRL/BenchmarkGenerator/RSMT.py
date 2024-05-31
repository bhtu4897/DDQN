
#pip install pyRSMT
import numpy as np
import pyrsmt

def generateRSMT(twoPinList):
    # Two pin list: [[(0,0,0),(2,0,0)],[(2,0,0),(2,0,1)],[(2,0,1),(2,2,0)]]
    # Generate sorted two pin list based on rectilinear steiner minimum tree approach

    # 提取所有引腳點
    pinList = []
    for i in range(len(twoPinList)):
        pinList.append(twoPinList[i][0])
        pinList.append(twoPinList[i][1])

    # 去除重複的引腳點
    pinList = list(set(pinList))

    # 將三維引腳轉換為二維引腳，因為RSMT通常處理二維平面
    pinList2D = [(x, y) for (x, y, z) in pinList]

    # 使用pyRSMT計算RSMT
    rsmt = pyrsmt.RSMT(pinList2D)

    # 將RSMT結果轉換為兩引腳列表
    twoPinListSorted = []
    for edge in rsmt.edges:
        p1, p2 = edge
        # 需要將二維點映射回三維點
        z1 = next(p[2] for p in pinList if (p[0], p[1]) == p1)
        z2 = next(p[2] for p in pinList if (p[0], p[1]) == p2)
        twoPinListSorted.append([(p1[0], p1[1], z1), (p2[0], p2[1], z2)])

    return twoPinListSorted

if __name__ == '__main__':
    # 測試生成RSMT
    twoPinList = [[(0,0,0),(2,0,0)],[(2,0,0),(2,0,1)],[(2,0,1),(2,2,0)],[(2,2,0),(0,0,1)]]
    RSMT = generateRSMT(twoPinList)
    print('Sorted Two pin list: ', RSMT)
