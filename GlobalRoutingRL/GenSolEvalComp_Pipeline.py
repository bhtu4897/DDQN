import os
import argparse

def parse_arguments():  #argparse: 用於解析命令行參數。
    parser = argparse.ArgumentParser('Benchmark Generator Parser') #創建一個命令行參數解析器。
    parser.add_argument('--benchNumber',type=int,\
        dest='benchmarkNumber',default=20)
    parser.add_argument('--gridSize',type=int,dest='gridSize',default=16)
    parser.add_argument('--netNum',type=int,dest='netNum',default=5)
    parser.add_argument('--capacity',type=int,dest='cap',default=4)
    parser.add_argument('--maxPinNum',type=int,dest='maxPinNum',default=5)
    parser.add_argument('--reducedCapNum',type=int,dest='reducedCapNum',default=1)

    return parser.parse_args()


if __name__ == '__main__':
	# Remember to copy results to other directory when running new parameters 記得在運行新參數時將結果複製到其他目錄

	filename = None
	args = parse_arguments()
	benNum = args.benchmarkNumber
	gridSize = args.gridSize; netNum = args.netNum
	cap = args.cap; maxPinNum = args.maxPinNum
	reducedCapNum = args.reducedCapNum

	# Generating problems module (A*)
	# Make sure previous benchmark files: "benchmark","capacityplot",
	# and 'solution' are removed
	# 生成問題模塊 (A*)
    # 確保之前的基準文件："benchmark", "capacityplot", 和 'solution' 被刪除
	os.system('rm -r benchmark')  #os: 用於執行操作系統命令和改變工作目錄
	os.system('rm -r capacityplot_A*')
	os.system('rm -r solutionsA*')
	os.system('rm -r solutionsDRL')
	os.chdir('BenchmarkGenerator') #切換工作目錄
	# os.chdir('..')
	print('**************')
	print('Problem Generating Module')
	print('**************')
	os.system('python BenchmarkGenerator.py --benchNumber {benNum} --gridSize {gridSize}\
	 --netNum {netNum} --capacity {cap} --maxPinNum {maxPinNum} --reducedCapNum {reducedCapNum}'\
	 .format(benNum=benNum,gridSize=gridSize,\
	 	netNum=netNum,cap=cap,maxPinNum=maxPinNum,reducedCapNum=reducedCapNum))   # 運行 BenchmarkGenerator.py

	# Solve problems with DRL 用DRL解決問題
	os.chdir('..') # Go back to main folder 返回主文件夾
	os.system('python Router.py') # 運行 Router.py
	# !!! Code the results for heatmap with DRL solution
    # !!! 使用深度強化學習 (DRL) 解決方案生成熱圖
    

	# Evaluation of DRL and A* solution
    # 評估 DRL 和 A* 解決方案
    

	# Plot results 
	# WL and OF with sorted A* results; difference in WL
    # 繪製結果
    # 用排序的 A* 結果來比較等待時間 (WL) 和目標函數 (OF)；以及 WL 的差異

	

