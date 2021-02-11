"""
柔性作业车间调度问题
(Flexible Job-shop Scheduling Problem, FJSP)
农历旧年的最后一天，老娘以为我是为挣那两个加班费才选择在年三十这天还工作，实际只是因为这次的年没有外婆。
以前只是年味减少，现在对我来说是彻底没了年。
二十六年来，从开始记事起每年三十都在外婆身边。哪怕父母去爷爷家和我分开过年，我也执意留在外婆家。
十来岁时和外婆年前赶集，她总会由我的喜好给我买上一堆玩意儿，然后在我妈面前护着我，“就过年玩这两天，你别说他！”，
年龄稍长后不贪玩了，年前便陪着她炸圆子包饺子看春晚，“你以后工作可别跑那么远，就待在合肥，没事还能送回来给我看两眼”
最早体会到分别的痛苦是六岁。在外婆家度过了无忧无虑的童年，随父母去上海。临行前与外婆抱一起哭成泪人，她答应我两个月后会去看我。
我问我妈两个月是多久，“是8周”，一周又是多久，“是7天，一天24小时”，也就那会儿我算术特别好。
后来每次寒暑假欢欢喜喜见面，又泪流满面的分别。
五年级往后再和外婆分别便不会再哭了，而外婆依旧每次都会流眼泪。那时我还无法理解她为何如此感性，有啥可哭的，又不是不会再见面了...
所以外婆离开那天我也没哭...
外婆爱打牌，性格固执不听子女的话，哪怕是刚化疗结束也要去麻将馆，为随她的喜好，在她临走的那天我买了一副麻将随她一起。
外婆还爱喝酒，但从不喝多，夏天啤酒冬天白酒，我四五岁时，外婆倒满一碗啤酒，会让我把啤酒沫吸掉。我爱喝酒也是那时培养出来的。
"""
import numpy as np
import random
from typing import List
from matplotlib import pyplot as plt
import heapq  as he

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# LOT数，统一站点数，机台数
job_num = 5
process_num = 2
machine_num = 6
# job_num个LOT的process_num个站点在machine_num台机台上的加工时间
times = [
   [
        [2,2,None,2,None,None],
        [None,None,None,None,2,None]
   ],
   [
        [1,None,1,None,None,None],
        [None,None,None,None,3,3]
   ],
   [
        [None,2,2,None,None,None],
        [None,None,None,None,2,None]
   ],
   [
        [3,None,None,3,None,None],
        [None,None,None,None,4,4]
   ],
   [
        [None,3,None,3,None,None],
        [None,None,None,None,2,2]
   ]
]
# qtime约束list，999999表示该LOT没有qtime约束，其他数值表示qtime约束时间
qtime=[0,0,0,0,0]

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
        # 把这个位置可用的jLOT的信息素浓度求和
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
    :return: 计算出的总时间花费，每个process加工完后机器的时间，每个process的机器选择
    """
    # 记录每个LOT在拓扑序中出现的次数，以确定是第几个工序
    job_use = [0 for _ in range(job_num)]
    # 记录每个process加工完后机器的时间
    process_times = [
        [
            [0 for _ in range(machine_num)] 
            for _ in range(process_num)
        ]
        for _ in range(job_num)
    ]
    #记录每个process的机器选择
    machine_choice=[
        [0 for _ in range(process_num)]
        for _ in range(job_num)
    ]
    
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
        # 将每个process的机器号写入machine_choice
        machine_choice[job_id][process_id]=machine_id 
        # 获取max(该LOT上一站点时间,该machine上一任务完成时间)
        start_time[job_id][process_id] = max(job_end_times[job_id], machine_end_times[machine_id])
        # 计算当前结束时间，写入这两个表
        job_end_times[job_id] =machine_end_times[machine_id] =start_time[job_id][process_id] + times[job_id][process_id][machine_id]
        # 将machine_end_times写入process_times
        for m in range(machine_num):
            process_times[job_id][process_id][m]=machine_end_times[m]
        # 维护job_use
        job_use[job_id] += 1
        
    return max(job_end_times),process_times,machine_choice,start_time

def diff_time(topo_jobs: List[int], process2machine: List[List[int]]):
    """
    计算同一LOT不同站点之间的时间差
    :return: 时间差
    """
    # 记录同一LOT不同step之间的时间差(qtime)
    process_timediff=[0 for _ in range(job_num)]
    # 生成每个process加工完后机器的时间
    times_process = cal_time(topo_jobs, process2machine)[1]
    # 生成每个process的机器选择
    choice_machine = cal_time(topo_jobs, process2machine)[2] 
    for job_id in range(job_num):
        process_id = 0
        process_timediff[job_id]=times_process[job_id][process_id+1][choice_machine[job_id][process_id+1]]- times_process[job_id][process_id][choice_machine[job_id][process_id]]-times[job_id][process_id+1][choice_machine[job_id][process_id+1]]
    return process_timediff

def if_else(topo_jobs: List[int], process2machine: List[List[int]]):
    """
    判断实际run货是否在qtime约束范围内
    :return: 在qtime约束范围内的LOT个数；
             对于超过qtime约束的LOT序号，计算qtime与qtime约束之间的差值总和
    """
    # 记录相较于qtime约束后的LOT信息
    ifelse = [0 for _ in range(job_num)]
    # 计算不同step之间的时间差
    timediff = diff_time(topo_jobs, process2machine)
    # 对于超过qtime约束的LOT,计算qtime与qtime约束之间的时间差总和
    totaldiff_qtime = 0
    # 记录同一LOT，没有超过qtime约束记为None，超过qime约束的记为1
    for job_id in range(job_num):
        if timediff[job_id] <= qtime[job_id]:
           ifelse[job_id]=None
        else:
           ifelse[job_id]=1  
    # 获取超过qtime约束的LOT序号(ifelse里不为None)
    job_over = [job_id for job_id in range(job_num)
                if ifelse[job_id]is not None]  
    # 对于超过qtime约束的LOT，计算qtime与qtime约束之间的差值总和
    for job_id in job_over:
        totaldiff_qtime += timediff[job_id]-qtime[job_id]
    return ifelse.count(None),totaldiff_qtime



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
bestcount = 0
best_diff = 999999
best_time = 999999
best_totaldiff = [0 for _ in range(job_num)]
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
    # 记录每只蚂蚁所选路径在Qtime约束范围内的LOT数量
    incount = [0 for _ in range(ant_num)]
    # 记录每只蚂蚁所选路径在Qtime约束范围外的LOT,实际Qtime与Qtime约束的差值总和
    qdiff = [0 for _ in range(ant_num)]
    # 记录每只蚂蚁所选路径run完所有LOT所花费的总时间
    timesummary = [0 for _ in range(ant_num)]
    # 记录每只蚂蚁所选路径产生的不同step之间的时间差 
    totaldiff = [
        [-1 for _ in range(job_num)]
        for _ in range(ant_num)
    ]
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
        # 计算不同step之间的时间差(Qtime)
        difftime = diff_time(topo_jobs, process2machine)
        # 记录每只蚂蚁经过的topo_jobs
        for job_id in range(job_num * process_num):
            total_topo_jobs[ant_id][job_id] = topo_jobs[job_id]
        # 记录每只蚂蚁经过的process2machine    
        for job_id in range(job_num):
            for process_id in range(process_num):
                total_process2machine[ant_id][job_id][process_id] = process2machine[job_id][process_id]
        # 记录每只蚂蚁所选路径产生的不同step之间的时间差          
        for job_id in range(job_num):
            totaldiff[ant_id][job_id] = difftime[job_id]
        # 记录每只蚂蚁所选路径在Qtime约束范围内的LOT数量
        incount[ant_id] = if_else(topo_jobs, process2machine)[0]
        # 记录每只蚂蚁所选路径实际Qtime与Qtime约束的差值总和
        qdiff[ant_id] = if_else(topo_jobs, process2machine)[1]
        # 记录每只蚂蚁所选路径run完所有LOT所花费的总时间
        timesummary[ant_id] = cal_time(topo_jobs, process2machine)[0]
        # 记录每只蚂蚁所选路径每批LOT每个站点开始加工的时间
        total_start_time[ant_id] = cal_time(topo_jobs, process2machine)[3]
        
    # 统计所有蚂蚁所选的路径中在Qtime约束范围内的LOT数量最大的是几批LOT
    maxincount = max(incount)
    # 统计所有蚂蚁所选的路径中在Qtime约束范围内的LOT数量最大的个数
    maxcount = 0
    for ant_id in range(ant_num):
        if incount[ant_id] == maxincount:
            maxcount +=1 
    # 统计incount中最大的数，即最大的在Qtime约束范围内的LOT数量
    max_number = he.nlargest(maxcount,incount)
    # 用于记录incount中最大数的索引号
    max_index = []
    for t in max_number:
        index = incount.index(t)
        max_index.append(index)
        incount[index] = 999999
    
    # 用于储存与Qtime约束的差值总和，个数与incount中最大数的个数相同，即为相对应的差值总和
    q_diff = [0 for _ in range(maxcount)]
    for t in range(maxcount):
        q_diff[t]=qdiff[max_index[t]]
        
    # 统计q_diff中最小的值个数（与Qtime约束的差值总和最小）
    minq_diff = 0
    for t in range(maxcount):
        if q_diff[t] == min(q_diff):
            minq_diff +=1
    # 统计q_diff中最小的数，即与Qtime约束的差值总和的最小值，个数即是minq_diff
    min_number = he.nsmallest(minq_diff,q_diff)
    # 用于记录q_diff中最小数的索引号
    min_index = []
    for t in min_number:
        index = q_diff.index(t)
        min_index.append(index)
        q_diff[index] = 999999
    
    # 用于储存总时间，个数与q_diff中最小的值个数相同
    time_summary = [0 for _ in range(minq_diff)]
    for t in range(minq_diff):
        time_summary[t]=timesummary[max_index[min_index[t]]]
    # 统计time_summary中最小的数，即最小的总时间
    min_number1 = he.nsmallest(1,time_summary)
    # 用于记录time_summary中最小数的索引号
    min_index1 = []
    for t in min_number1:
        index = time_summary.index(t)
        min_index1.append(index)
        time_summary[index] = 999999
    
         
    # 用于更新最优的拓扑序和机器分配
    if (maxincount > bestcount) or ((maxincount == bestcount) and (qdiff[max_index[min_index[min_index1[0]]]] < best_diff)) or ((maxincount == bestcount) and (qdiff[max_index[min_index[min_index1[0]]]] == best_diff) and (timesummary[max_index[min_index[min_index1[0]]]] < best_time)):
        bestcount = maxincount 
        best_diff = qdiff[max_index[min_index[min_index1[0]]]]
        best_time = timesummary[max_index[min_index[min_index1[0]]]]
        for d in range(job_num * process_num):
            best_topo_jobs[d] = total_topo_jobs[max_index[min_index[min_index1[0]]]][d]
        for job_id in range(job_num):
            for process_id in range(process_num):
                best_process2machine[job_id][process_id] = total_process2machine[max_index[min_index[min_index1[0]]]][job_id][process_id]
        for job_id in range(job_num):
            best_totaldiff[job_id] = totaldiff[max_index[min_index[min_index1[0]]]][job_id]
        for job_id in range(job_num):
            for process_id in range(process_num):
                best_start_time[job_id][process_id] = total_start_time[max_index[min_index[min_index1[0]]]][job_id][process_id]
        
        
        
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
for o in range(machine_num):
    print("M{}\t".format(o + 1), end='')
    print(np.round((best_machinetime[o]),2),end='\t')
    print(np.round((best_machinetime[o])/60,2),end='\n')

  
print("\n\t\t[迭代时间]")
print(np.round(time_list,2))
 
print("\n\t\t[收敛时间]")
print(np.round(max(time_list, key=time_list .count),2))

print("\n\t\t[Qtime]",end='\n')
print(best_totaldiff, end='\n')

print("\n\t\t[在Qtime约束范围内的LOT数量]")
print(bestcount, end='\n')

print("\n\t\t[超过Qtime约束的LOT,Qtime与Qtime约束的差值总和]")
print(best_diff, end='\n')

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
    "size": 9
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
        plt.text(best_start_time[job_id][process_id],best_process2machine[job_id][process_id]-0.25, 'step%s'%(process_id+1), fontdict=fontdict_lot)
ylabels = []  # 生成y轴标签
for machine_id in range(machine_num):
    ylabels.append('M' + str(machine_id+1))
plt.yticks(range(machine_num), ylabels, rotation=0)
plt.title("机台派工甘特图")
plt.xlabel("加工时间")
plt.ylabel("机台")
plt.show()


# 定义字体，大小（机台派工甘特图） 
fontdict_machine = {
    "family": "Microsoft YaHei",
    "style": "oblique",
    "color": "black",
    "size": 12
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


