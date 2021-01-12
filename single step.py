"""
柔性作业车间调度问题
(Flexible Job-shop Scheduling Problem, FJSP)
"""
import numpy as np
import random
from typing import List
from matplotlib import pyplot as plt
import heapq  as he

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 作业数，统一工序数，机器数
job_num = 60
process_num = 1
machine_num = 10
# job_num个Job的process_num个工序在machine_num台机器上的加工时间
times = [
[[2,2,None,2,None,2,None,2,None,None]],
[[2,None,2,None,None,2,None,2,None,None]],
[[None,None,None,3,None,None,3,None,3,None]],
[[None,2,2,None,2,None,None,None,2,None]],
[[2,None,2,None,None,2,2,None,None,2]],
[[None,None,None,None,3,None,None,None,3,None]],
[[None,None,4,None,None,4,None,4,None,4]],
[[3,3,None,3,None,None,3,None,3,None]],
[[2,None,None,None,None,None,2,None,2,2]],
[[None,None,3,None,3,None,None,None,3,None]],
[[None,None,4,4,None,4,None,None,4,4]],
[[None,2,None,None,2,None,None,None,2,None]],
[[4,None,4,None,None,None,4,4,None,None]],
[[None,None,2,None,2,None,None,2,None,None]],
[[None,3,None,3,None,3,None,None,3,None]],
[[None,None,None,4,None,None,4,None,4,None]],
[[None,4,None,4,None,4,None,None,4,None]],
[[3,None,3,None,None,None,None,None,3,3]],
[[None,2,None,2,None,2,None,None,None,2]],
[[None,None,4,None,None,4,None,4,4,None]],
[[3,3,None,None,None,3,None,3,None,3]],
[[None,3,None,3,None,None,3,None,3,None]],
[[2,None,2,None,2,None,None,None,None,2]],
[[None,4,None,4,None,None,None,4,4,None]],
[[5,None,None,5,None,5,None,5,None,5]],
[[None,2,None,2,None,2,2,None,None,None]],
[[3,None,3,3,None,None,None,None,3,None]],
[[None,4,None,4,None,None,None,None,4,4]],
[[None,None,None,5,None,5,5,None,None,5]],
[[None,None,3,None,3,None,None,3,3,None]],
[[2,2,None,2,None,2,None,2,None,None]],
[[2,None,2,None,None,2,None,2,None,None]],
[[None,None,None,3,None,None,3,None,3,None]],
[[None,2,2,None,2,None,None,None,2,None]],
[[2,None,2,None,None,2,2,None,None,2]],
[[None,None,None,None,3,None,None,None,3,None]],
[[None,None,4,None,None,4,None,4,None,4]],
[[None,3,None,None,3,None,3,None,3,None]],
[[2,None,None,None,None,None,2,None,2,2]],
[[None,None,3,None,3,None,None,None,3,None]],
[[None,None,4,4,None,4,None,None,4,4]],
[[None,2,None,None,2,None,None,None,2,None]],
[[4,None,4,None,None,None,4,4,None,None]],
[[None,None,2,None,2,None,None,2,None,None]],
[[None,3,None,3,None,3,None,None,3,None]],
[[None,None,None,4,None,None,4,None,4,None]],
[[None,4,None,4,None,4,None,None,4,None]],
[[3,None,3,None,None,None,None,None,3,3]],
[[None,2,None,2,None,2,None,None,None,2]],
[[None,None,4,None,None,4,None,None,4,4]],
[[None,3,None,None,None,3,None,None,None,3]],
[[None,3,None,None,3,None,3,None,None,None]],
[[2,None,2,None,2,None,None,None,None,2]],
[[None,4,None,4,None,None,None,4,4,None]],
[[5,None,None,None,5,None,None,None,None,5]],
[[None,2,None,None,None,2,None,None,None,None]],
[[3,None,3,3,None,None,None,None,None,3]],
[[None,4,None,4,None,None,None,None,4,4]],
[[None,None,None,5,5,None,None,None,None,5]],
[[None,None,3,None,3,None,None,3,None,3]]
]
# 拓扑序的信息素浓度，初始值100
topo_phs = [
    [100 for _ in range(job_num)]
    for _ in range(job_num * process_num)
]

def gen_topo_jobs() -> List[int]:
    """
    生成拓扑序
    LOT在时空上处理的的拓扑序(Job索引)，这个序不能体现工序选择的机器a
    :return 如[0,1,0,2,2,...]表示p11,p21,p12,p31,p32,...
    """
    # 按照每个位置的信息素浓度加权随机给出
    # 返回的序列长，是LOT数量*站点的数量
    len = job_num * process_num
    # 返回的序列，最后这些-1都会被设置成0~job_num-1之间的索引
    ans = [-1 for _ in range(len)]
    # 记录每个LOT使用过的次数，用来防止LOT被使用超过process_num次
    job_use = [0 for _ in range(job_num)]
    # 记录现在还没超过process_num因此可用的job_id，每次满了就将其删除
    job_free = [job_id for job_id in range(job_num)]
    # 对于序列的每个位置
    for i in range(len):
        # 把这个位置可用的LOT的信息素浓度求和
        ph_sum = np.sum(list(map(lambda j: topo_phs[i][j], job_free)))
        # 接下来要随机在job_free中取一个job_id
        # 但是不能直接random.choice，要考虑每个LOT的信息素浓度
        test_val = .0
        rand_ph = random.uniform(0, ph_sum)
        for job_id in job_free:
            test_val += topo_phs[i][job_id]
            if rand_ph <= test_val:
                # 将序列的这个位置设置为job_id，并维护job_use和job_free
                ans[i] = job_id
                job_use[job_id] += 1
                if job_use[job_id] == process_num:
                    job_free.remove(job_id)
                break
    return ans

machine_phs = [
    [
        [100 for _ in range(machine_num)]
        for _ in range(process_num)
    ]
    for _ in range(job_num)
]

def gen_process2machine() -> List[List[int]]:
    """
    生成每个LOT的每个工序对应的机器索引号矩阵
    :return: 二维int列表，如[0][0]=3表示Job1的p11选择机器m4
    """
    # 要返回的矩阵，共job_num行process_num列，取值0~machine_num-1
    ans = [
        [-1 for _ in range(process_num)]
        for _ in range(job_num)
    ]
    # 对于每个位置，也是用信息素加权随机出一个machine_id即可
    for job_id in range(job_num):
        for process_id in range(process_num):
            # 获取该位置的所有可用机器号(times里不为None)
            machine_free = [machine_id for machine_id in range(machine_num)
                            if times[job_id][process_id][machine_id] is not None]
            # 计算该位置所有可用机器的信息素之和
            ph_sum = np.sum(list(map(lambda m: machine_phs[job_id][process_id][m], machine_free)))
            # 还是用随机数的方式选取
            test_val = .0
            rand_ph = random.uniform(0, ph_sum)
            for machine_id in machine_free:
                test_val += machine_phs[job_id][process_id][machine_id]
                if rand_ph <= test_val:
                    ans[job_id][process_id] = machine_id
                    break
    return ans

def cal_time(topo_jobs: List[int], process2machine: List[List[int]]):
    """
    给定拓扑序和机器索引号矩阵
    :return: 计算出的总时间花费，每个process加工完后机器的时间，每批lot每个站点的开始时间
    """
    # 记录每个LOT在拓扑序中出现的次数，以确定是第几个工序
    job_use = [0 for _ in range(job_num)]

    #记录每批LOT每个站点的开始时间
    start_time=[
        [0 for _ in range(process_num)]
        for _ in range(job_num)
    ]
    # 循环中要不断查询和更新这两张表
    # (1)每个machine上一道工序的结束时间
    machine_end_times = [0 for _ in range(machine_num)]
    # (2)每个工件上一道工序的结束时间
    job_end_times = [0 for _ in range(job_num)]
    # 对拓扑序中的每个job_id
    for job_id in topo_jobs:
        # 在job_use中取出工序号
        process_id = job_use[job_id]
        # 在process2machine中取出机器号
        machine_id = process2machine[job_id][process_id]
        # 获取max(该LOT上一站点时间,该machine上一任务完成时间)
        start_time[job_id][process_id] = max(job_end_times[job_id], machine_end_times[machine_id])
        # 计算当前结束时间，写入这两个表
        job_end_times[job_id] =machine_end_times[machine_id] =start_time[job_id][process_id] + times[job_id][process_id][machine_id]
        # 维护job_use
        job_use[job_id] += 1
        
    return max(job_end_times),start_time


# 迭代次数
iteration_num = 200

# 蚂蚁数量
ant_num = machine_num*3                                                                      

# 绘图用
iter_list = range(iteration_num)
time_list = [0 for _ in iter_list]

best_topo_jobs = [-1 for _ in range(job_num * process_num)]
best_process2machine = [
    [-1 for _ in range(process_num)]
    for _ in range(job_num)
]

best_machinetime= [0 for _ in range(machine_num)]
best_time = 999999
best_start_time = [
    [0 for _ in range(process_num)]
    for _ in range(job_num)
]

# 对于每次迭代
for it in iter_list:
    # 记录每只蚂蚁经过的topo_jobs
    total_topo_jobs = [
        [-1 for _ in range(job_num * process_num)]
        for _ in range(ant_num)
    ]
    # 记录每只蚂蚁经过的process2machine（每只蚂蚁每个站点选择哪个机台）
    total_process2machine = [
        [
            [-1 for _ in range(process_num)]
            for _ in range(job_num)
                
        ]
            for _ in range(ant_num)
    ]
    # 记录每只蚂蚁所选路径run完所有LOT所花费的总时间
    timesummary = [0 for _ in range(ant_num)]

    # 记录每只蚂蚁所选路径不同站点的开始时间
    total_start_time = [
        [
            [0 for _ in range(process_num)]
            for _ in range(job_num)
        ]
            for _ in range(ant_num)
    ]
    # 对于每只蚂蚁
    for ant_id in range(ant_num):
        # 生成拓扑序
        topo_jobs = gen_topo_jobs()
        # 生成每道工序的分配机器索引号矩阵
        process2machine = gen_process2machine()
        # 记录每只蚂蚁经过的topo_jobs
        for job_id in range(job_num * process_num):
            total_topo_jobs[ant_id][job_id] = topo_jobs[job_id]
        # 记录每只蚂蚁经过的process2machine    
        for job_id in range(job_num):
            for process_id in range(process_num):
                total_process2machine[ant_id][job_id][process_id] = process2machine[job_id][process_id]
        # 记录每只蚂蚁所选路径run完所有LOT所花费的总时间
        timesummary[ant_id] = cal_time(topo_jobs, process2machine)[0]
        # 记录每只蚂蚁所选路径每批LOT每个站点开始加工的时间
        total_start_time[ant_id] = cal_time(topo_jobs, process2machine)[1]
     
   
    # 统计最小总时间
    mintime = min(timesummary)
    # 统计最小总时间的个数
    mintimecount = 0
    for ant_id in range(ant_num):
        if timesummary[ant_id] == mintime:
            mintimecount +=1 

    # 统计time_summary中最小的数，即最小的总时间
    min_number = he.nsmallest(1,timesummary)
    # 用于记录time_summary中最小数的索引号
    min_index = []
    for t in min_number:
        index = timesummary.index(t)
        min_index.append(index)
        timesummary[index] = 999999
   
         
    # 用于更新最优的拓扑序和机器分配
    if mintime < best_time:
        best_time = mintime
        for d in range(job_num * process_num):
            best_topo_jobs[d] = total_topo_jobs[min_index[0]][d]
        for job_id in range(job_num):
            for process_id in range(process_num):
                best_process2machine[job_id][process_id] = total_process2machine[min_index[0]][job_id][process_id]
        for job_id in range(job_num):
            for process_id in range(process_num):
                best_start_time[job_id][process_id] = total_start_time[min_index[0]][job_id][process_id]
      
        
        # 更新信息素浓度更新拓扑序信息素浓度表，更新每个Job的每个工序的信息素浓度表
        for i in range(job_num * process_num):
            for j in range(job_num):
                if j == best_topo_jobs[i]:
                    topo_phs[i][j] *= 1.1
                else:
                    topo_phs[i][j] *= 0.9
        for j in range(job_num):
            for p in range(process_num):
                for m in range(machine_num):
                    if m == best_process2machine[j][p]:
                        machine_phs[j][p][m] *= 1.1
                    else:
                        machine_phs[j][p][m] *= 0.9
                    
    time_list[it] = best_time 

# 输出解
print("\n\t\t[LOT分配给机台的情况]")
print("\t", end='')
for machine_id in range(machine_num):
    print("\tM{}".format(machine_id + 1), end='')
print()
for job_id in range(job_num):
    for process_id in range(process_num):
        print("P{}{}\t".format(job_id + 1,process_id+1), end='')
        for machine_id in range(machine_num):
            if machine_id == best_process2machine[job_id][process_id]:
                print("\t√", end='')
            else:
                print("\t-", end='')
        print("")

print("\n\t\t[LOT分配给机台的顺序]")
job_use = [0 for _ in range(job_num)]
for job_id in best_topo_jobs:
    print("P{}{} ".format(job_id + 1,job_use[job_id]+1), end='')
    job_use[job_id] += 1

for j in range(job_num):
        for p in range(process_num):
            for m in range(machine_num):
                if m == best_process2machine[j][p]:
                   best_machinetime[m]+=times[j][p][m]
                else:
                   best_machinetime[m]+=0
                   

print("\n\n\t\t[机台加工时间]")
for machine_id in range(machine_num):
    print("M{}\t".format(machine_id + 1), end='')
    print(np.round((best_machinetime[machine_id]),2),end='\t')
    print(np.round((best_machinetime[machine_id])/60,2),end='\n')

  
print("\n\t\t[迭代时间]")
print(np.round(time_list,2))
 
print("\n\t\t[收敛时间]")
print(np.round(max(time_list, key=time_list .count),2))


# 绘图（迭代收敛图）
plt.plot(iter_list, time_list)
plt.xlabel("迭代轮次")
plt.ylabel("时间")
plt.title("蚁群算法")
plt.show()

# 定义字体，大小（机台派工甘特图） 
fontdict_lot = {
    "family": "Microsoft YaHei",
    "style": "oblique",
    "color": "black",
    "size": 8
}
# 定义color(R)
color1 = np.random.randint(0,255,job_num)
# 定义color(G)
color2 = np.random.randint(0,255,job_num)
# 定义color(B)
color3 = np.random.randint(0,255,job_num)
# 绘图（机台派工甘特图）    
for job_id in range(job_num):
    for process_id in range(process_num):
        plt.barh(y=best_process2machine[job_id][process_id], width=times[job_id][process_id][best_process2machine[job_id][process_id]], left=best_start_time[job_id][process_id], edgecolor="black", color=(color1[job_id]/255,color2[job_id]/255,color3[job_id]/255))
        plt.text(best_start_time[job_id][process_id],best_process2machine[job_id][process_id], 'LOT%s'%(job_id+1), fontdict=fontdict_lot)
  
ylabels = []  # 生成y轴标签
for machine_id in range(machine_num):
    ylabels.append('M' + str(machine_id+1))
plt.yticks(range(machine_num), ylabels, rotation=0)
plt.title("机台派工甘特图")
plt.xlabel("加工时间")
plt.ylabel("机台")
plt.show()


# 定义字体，大小（LOT加工甘特图） 
fontdict_machine = {
    "family": "Microsoft YaHei",
    "style": "oblique",
    "color": "black",
    "size": 8
}
# 定义color(R)
color1_machine =np.random.randint(0,255,machine_num)
# 定义color(G)
color2_machine =np.random.randint(0,255,machine_num)
# 定义color(B)
color3_machine =np.random.randint(0,255,machine_num)
# 绘图（LOT加工甘特图）    
for job_id in range(job_num):
    for process_id in range(process_num):
        plt.barh(y=job_id, width=times[job_id][process_id][best_process2machine[job_id][process_id]], left=best_start_time[job_id][process_id], edgecolor="black", color=(color1_machine[best_process2machine[job_id][process_id]]/255,color2_machine[best_process2machine[job_id][process_id]]/255,color3_machine[best_process2machine[job_id][process_id]]/255))
        plt.text(best_start_time[job_id][process_id],job_id, 'M%s'%(best_process2machine[job_id][process_id]+1), fontdict=fontdict_machine)
# 生成y轴标签
ylabels = []  
for machine_id in range(job_num):
    ylabels.append('LOT' + str(machine_id+1))
plt.yticks(range(job_num), ylabels, rotation=0)
plt.title("LOT加工甘特图")
plt.xlabel("加工时间")
plt.ylabel("LOT")
plt.show()
