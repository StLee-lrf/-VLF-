import numpy as np
import matplotlib.pyplot as plt

#等离子体背景

"""
##
##

声明：这里的函数单位是千米，同时密度单位是无量纲的，因为这里的密度是归一化的
nespec: 等离子体物种数目 = 4

我们可以调整的参数有：
hwpp: 等离子体层边界厚度
therm: 等离子体温度
hwli: 下层电离层的厚度
rrli: 下层电离层的参考半径
n_op: 背景密度
fkc: 等离子体频率

##
##
"""


def densao(path, xlpp):
    """
    使用 Aikyo-Ondo (AO) 模型计算等离子体密度及其梯度。

    参数：
    - path: 包含 [半径（km）, 余纬度（度）, 经度（度）] 的列表或 numpy 数组。
    - re: 地球半径（单位：km）。
    - xlpp: 等离子体层边界的径向位置（单位：地球半径）。

    返回值：
    - n_ao: 等离子体密度（无量纲，归一化）。
    - dnadrr: 等离子体密度的径向梯度。
    - dnadth: 等离子体密度的纬度梯度。
    - dnadph: 等离子体密度的经度梯度（AO 模型中始终为 0）。
    """

##基本参数
    hwpp = 0.1  # 等离子体层边界厚度
    rad = np.pi / 180.0  # 角度转弧度的系数
    re = 6370.0  # 地球半径 [km]
##基本参数

    # 从路径中提取位置参数
    rr = path[0]  # 半径 [km]
    th = path[1]  # 余纬度 [度]
    ph = path[2]  # 经度 [度]

    # 计算三角函数值
    costh = np.cos(rad * th)  # 余纬度角的余弦值
    sinth = np.sin(rad * th)  # 余纬度角的正弦值
    cotth = 1.0 / np.tan(rad * th)  # 余纬度角的余切值

    # 计算归一化的等离子体路径变量 (XLV)
    xlv = rr / re / sinth**2  # XLV 表示当前路径的归一化变量
    dlv_drr = xlv / rr  # XLV 对半径的导数
    dlv_dth = -xlv * cotth * 2.0  # XLV 对余纬度的导数
    dlv_dph = 0.0  # XLV 对经度的导数（假设经度方向上均匀分布）

    # 根据等离子体层边界和厚度对 XLV 进行归一化
    xlvnrm = (xlv - xlpp) / hwpp  # 归一化的 XLV，用于描述边界变化

    # 使用 AO 模型计算等离子体密度 (XNAO)
    n_ao = 0.5 - np.arctan(xlvnrm) / np.pi  # 等离子体密度，随路径平滑变化

    # 计算等离子体密度梯度的中间变量
    df1_drr = dlv_drr / (1.0 + xlvnrm**2) / hwpp / np.pi  # 径向梯度的中间结果
    df1_dth = dlv_dth / (1.0 + xlvnrm**2) / hwpp / np.pi  # 纬度梯度的中间结果
    df1_dph = dlv_dph / (1.0 + xlvnrm**2) / hwpp / np.pi  # 经度梯度的中间结果

    # 计算等离子体密度的最终梯度
    dnadrr = -df1_drr / n_ao  # 径向梯度
    dnadth = -df1_dth / n_ao  # 纬度梯度
    dnadph = -df1_dph / n_ao  # 经度梯度（AO 模型中为 0）

    return n_ao, dnadrr, dnadth, dnadph


def densde(path, nspec):
    """
    根据扩散平衡模型 (Diffusive Equilibrium, DE) 计算等离子体密度及其径向梯度。

    参数：
    - path: 包含 [径向距离（km）] 的列表或 numpy 数组。
    - nspec: 等离子体物种数目（如电子、质子、离子等）。

    返回值：
    - n_de: 各种物种的归一化密度。
    - dnddrr: 各种物种的径向密度梯度。
    """

##基本参数
    rref = 7370.0 # 参考半径 [km]
    eta = np.array([0, 0.152, 0.82, 0.025])  # 比例常数
    # 将参考半径从 km 转换为 m
    rref_m = rref * 1e3  # 转换为米 (m)
    mp = 1.6726485e-27  # 质子质量 (kg)
    gm = 3.98603e14      # 地球引力常数 (m^3/s^2)
    k = 1.380662e-23    # 玻尔兹曼常数 (J/K)
    therm = 1000.0       # 等离子体温度 (K)
    # 计算重力加速度 g(r)
    g_r = gm / rref_m**2
    # 计算各物种的扩散系数
    shinv_2 = mp * g_r / (k * therm) * 1.0e-3  # SHINV(2)
    shinv_3 = mp * 4.0 * g_r / (k * therm) * 1.0e-3  # SHINV(3)
    shinv_4 = mp * 16.0 * g_r / (k * therm) * 1.0e-3  # SHINV(4)
    shinv = np.array([0, shinv_2, shinv_3, shinv_4])  # 缩放因子
##基本参数

    # 提取径向距离
    rr = path[0]  # 径向距离 [km]

    # 计算扩散平衡模型的中间变量
    gph = rref * (rr - rref) / rr  # 广义高度参数 GPH
    dgphdr = (rref / rr)**2  # GPH 对径向的导数

    # 各种物种的密度贡献
    etexp2 = eta[1] * np.exp(-gph * shinv[1])
    etexp3 = eta[2] * np.exp(-gph * shinv[2])
    etexp4 = eta[3] * np.exp(-gph * shinv[3])

    # 累积和，用于归一化和导数计算
    sum1 = etexp2 + etexp3 + etexp4  # 密度的总贡献
    sum2 = (etexp2 * shinv[1] + etexp3 * shinv[2] + etexp4 * shinv[3])  # 梯度的总贡献

    # 计算归一化密度
    n_de = np.zeros(nspec)
    n_de[0] = np.sqrt(sum1)  # 总归一化密度
    n_de[1] = etexp2 / n_de[0]  # 第一物种的密度
    n_de[2] = etexp3 / n_de[0]  # 第二物种的密度
    n_de[3] = etexp4 / n_de[0]  # 第三物种的密度

    # 计算径向密度梯度
    dnddrr = np.zeros(nspec)
    dnddrr[0] = -dgphdr * sum2 / sum1 / 2.0  # 总密度梯度
    dnddrr[1] = -dgphdr * shinv[1] - dnddrr[0]  # 第一物种的径向梯度
    dnddrr[2] = -dgphdr * shinv[2] - dnddrr[0]  # 第二物种的径向梯度
    dnddrr[3] = -dgphdr * shinv[3] - dnddrr[0]  # 第三物种的径向梯度

    return n_de, dnddrr


def densli(path):
    """
    计算下层电离层 (Lower Ionosphere, LI) 的密度及其径向梯度。

    参数：
    - path: 包含 [径向距离 (km)] 的列表或 numpy 数组。

    返回值：
    - n_li: 下层电离层的密度 (无量纲)。
    - dnldrr: 下层电离层密度的径向梯度。
    """

##基本参数
    rrli = 6370.0 + 90.0  # 下层电离层的参考半径 (km)
    hwli = 140.0  # 下层电离层的厚度 (km)
##基本参数

    # 提取径向距离
    rr = path[0]  # 径向距离 (km)

    # 判断径向位置是否在下层电离层范围内
    if (rr - rrli) > 0:
        # 计算归一化高度
        altnrm = (rr - rrli) / hwli  # 归一化高度

        # 计算密度
        n_li = 1.0 - np.exp(-altnrm**2)  # 下层电离层的密度

        # 计算密度的径向梯度
        dnldrr = np.exp(-altnrm**2) * altnrm * 2.0 / hwli / n_li
        #因为纬度和经度梯度在下层电离层中为梯度较小，所以不需要计算
    else:
        # 当径向距离在下层电离层范围外时，密度和梯度为 0
        n_li = 0.0
        dnldrr = 0.0

    return n_li, dnldrr

def densop():
    """
    计算背景等离子体的密度及其梯度。

    参数：
    - xn0: 背景等离子体的数密度（常量）。

    返回值：
    - xnop: 背景密度（等于 xn0）。
    - dnodrr: 背景密度的径向梯度（为 0）。
    - dnodth: 背景密度的纬度梯度（为 0）。
    - dnodph: 背景密度的经度梯度（为 0）。
    假设背景密度在所有方向上均匀分布，因此所有方向上的梯度均为零
    """
    # 背景密度
    n_op = 50000.0 # 假设背景密度为常量

    # 背景密度的梯度
    dnodrr = 0.0  # 径向梯度
    dnodth = 0.0  # 纬度梯度
    dnodph = 0.0  # 经度梯度

    return n_op, dnodrr, dnodth, dnodph


def dens(path, nspec, xlpp):
    """
    计算总的等离子体密度及其梯度，整合来自扩散平衡、电离层、等离子体层和背景等离子体的贡献。

    参数：
    - path: 包含路径信息的列表或 numpy 数组（如径向距离等）。
    - nspec: 等离子体物种的数量。

    返回值：
    - n_s: 各物种的总密度。
    - dnsdrr: 各物种的径向密度梯度。
    - dnsdth: 各物种的纬度密度梯度。
    - dnsdph: 各物种的经度密度梯度。
    """

    # 从各模块计算密度和梯度
    n_de, dnddrr = densde(path, nspec)  # 扩散平衡
    n_li, dnldrr = densli(path)  # 下层电离层
    n_ao, dnadrr, dnadth, dnadph = densao(path, xlpp)  # 等离子体层
    xnop, dnodrr, dnodth, dnodph = densop()  # 背景密度

    # 初始化结果数组
    n_s = np.zeros(nspec)
    dnsdrr = np.zeros(nspec)
    dnsdth = np.zeros(nspec)
    dnsdph = np.zeros(nspec)

    # 计算总密度
    for i in range(nspec):
        n_s[i] = n_de[i] * n_li * n_ao * xnop

    # 计算总的密度梯度
    for i in range(nspec):
        dnsdrr[i] = dnddrr[i] + dnldrr + dnadrr + dnodrr  # 径向梯度
        dnsdth[i] = dnadth + dnodth  # 纬度梯度
        dnsdph[i] = dnadph + dnodph  # 经度梯度

    return n_s, dnsdrr, dnsdth, dnsdph


def dipole(path):
    """
    计算偶极磁场的参数，包括方向余弦、方向导数和磁场强度。
    
    参数：
    - path: 包含 [径向距离 RR（km）, 余纬度 TH（度）, 经度 PH（度）] 的数组。
    
    返回值：
    - YR, YT, YP: 磁场方向的余弦分量。
    - DYRDRR, DYTDRR, DYPDRR: 磁场方向对径向距离的导数。
    - DYRDTH, DYTDTH, DYPDTH: 磁场方向对余纬度的导数。
    - DYRDPH, DYTDPH, DYPDPH: 磁场方向对经度的导数。
    - BB: 磁场强度。
    - DBBDRR, DBBDTH, DBBDPH: 磁场强度对径向距离、余纬度和经度的导数。
    """

    ##基本参数
    rad = np.pi / 180.0  # 角度转弧度的系数
    ##基本参数

    # 定义地磁偶极矩常数 GMDM
    gmdm = 8.07 * 10**6  # 单位：T·km³

    # 从路径中提取参数
    rr = path[0]  # 径向距离 (km)
    th = path[1]  # 余纬度 (度)
    ph = path[2]  # 经度 (度)

    # 三角函数计算
    costh = np.cos(rad * th)  # 余纬度角的余弦
    sinth = np.sin(rad * th)  # 余纬度角的正弦
    cosths = 1.0 + 3.0 * costh**2  # 偶极场归一化因子

    # 磁场方向的余弦分量
    yr = -2.0 * costh / np.sqrt(cosths)  # 径向分量
    yt = -sinth / np.sqrt(cosths)        # 余纬度分量
    yp = 0.0                             # 经度分量

    # 磁场方向的导数（径向距离方向）
    dyrdrr = 0.0
    dytdrr = 0.0
    dypdrr = 0.0

    # 磁场方向的导数（余纬度方向）
    dyrdth = -2.0 / cosths * yt
    dytdth = 2.0 / cosths * yr
    dypdth = 0.0

    # 磁场方向的导数（经度方向）
    dyrdph = 0.0
    dytdph = 0.0
    dypdph = 0.0

    # 磁场强度
    bb = gmdm * np.sqrt(cosths) / rr**3

    # 磁场强度的导数
    dbbdrr = -3.0 / rr  # 对径向距离的导数
    dbbdth = -3.0 * sinth * costh / cosths  # 对余纬度的导数
    dbbdph = 0.0  # 对经度的导数

    return (yr, yt, yp, 
            dyrdrr, dytdrr, dypdrr, 
            dyrdth, dytdth, dypdth, 
            dyrdph, dytdph, dypdph, 
            bb, dbbdrr, dbbdth, dbbdph)


def ref(path, nspec, xlpp, fkc):
    """
    Python 实现的 REF 函数，用于更新路径中的磁场方向、介质密度和折射率。

    参数：
    - path: 当前路径信息，列表或 numpy 数组 [RR, TH, PH, KX, KY, KZ]
    - nspec: 等离子体物种数目
    - xlpp: 等离子体层边界的径向位置
    - fkc: 等离子体频率(单位：kHz)

    返回:
    - pmu0: 波矢模长
    - copsi: 波矢与磁场的方向余弦
    - copsis: copsi 的平方
    - sipsi: copsi 的正弦
    - sipsis: sipsi 的平方
    - psi_angl: 波矢与磁场的夹角
    - px: 介质参数
    - py: 介质参数
    - pz: 介质参数
    - pa1: 介质参数
    - pa2: 介质参数
    - pa3: 介质参数
    - pk1: 介质参数
    - pk2: 介质参数
    - pk4: 介质参数
    - pk5: 介质参数
    - pa: 介质参数
    - pb: 介质参数
    - pc: 介质参数
    - pd: 介质参数
    - pmu: 折射率
    - gmu: 折射率,修正后
    - pmus:
    """
    ##基本参数

    re = 6370.0  # 地球半径 [km]
    rad = np.pi / 180.0  # 角度转弧度的系数
    deg = 1.0 / rad # 弧度转角度的系数
    rref = 7370.0 # 参考半径 [km]
    eta = np.array([0, 0.152, 0.82, 0.025])  # 比例常数
    # 将参考半径从 km 转换为 m
    rref_m = rref * 1e3  # 转换为米 (m)
    me = 9.109e-31      # 电子质量 (kg)
    mp = 1.6726485e-27  # 质子质量 (kg)
    qe = 1.602e-19       # 电荷量 (C)
    ep = 8.854187817e-12 # 真空介电常数 (F/m)
    gm = 3.98603e14      # 地球引力常数 (m^3/s^2)
    k = 1.380662e-23    # 玻尔兹曼常数 (J/K)
    therm = 1000.0       # 等离子体温度 (K)
    # 计算重力加速度 g(r)
    g_r = gm / rref_m**2
    pf0 = np.array([
    qe**2 / (4.0 * np.pi**2 * ep * me),           # PF0(1) 电子
    qe**2 / (4.0 * np.pi**2 * ep * mp),           # PF0(2) 质子
    qe**2 / (4.0 * np.pi**2 * ep * (mp * 4.0)),   # PF0(3) 氦离子
    qe**2 / (4.0 * np.pi**2 * ep * (mp * 16.0))   # PF0(4) 氧离子
    ])  # 等离子体频率的平方（分种类）
    gf0 = np.array([
    -qe / (2.0 * np.pi * me) * 1.0e-3,           # GF0(1): 电子
    qe / (2.0 * np.pi * mp) * 1.0e-3,            # GF0(2): 质子
    qe / (2.0 * np.pi * (mp * 4.0)) * 1.0e-3,    # GF0(3): 氦离子 (4 * m_p)
    qe / (2.0 * np.pi * (mp * 16.0)) * 1.0e-3    # GF0(4): 氧离子 (16 * m_p)
    ])
    ##基本参数


    # 调用子程序 DIPOLE 和 DENS
    n_s, dnsdrr, dnsdth, dnsdph = dens(path, nspec, xlpp)
    yr, yt, yp, dyrdrr, dytdrr, dypdrr, dyrdth, dytdth, dypdth, dyrdph, dytdph, dypdph, bb, dbbdrr, dbbdth, dbbdph = dipole(path)
    # 计算 PX, PY, PZ 分量
    px = pf0[:nspec] * n_s[:nspec] / fkc**2
    py = gf0[:nspec] * bb / fkc
    pz = py**2 - 1.0

    # 计算波矢模长和方向与磁场的几何关系
    pmu0 = np.sqrt(path[3]**2 + path[4]**2 + path[5]**2) # 波矢模长
    copsi = (yr * path[3] + yt * path[4] + yp * path[5]) / pmu0 # 方向余弦
    copsi = np.clip(copsi, -1.0, 1.0)  # 限制值在 [-1, 1] 范围内
    copsis = copsi**2 # 方向余弦的平方
    sipsis = abs(1.0 - copsis) # 方向正弦的平方
    sipsi = np.sqrt(sipsis) # 方向余弦的正弦
    psi_angl = deg * np.arccos(copsi) # 波矢与磁场的夹角
    #记得输出


    # 初始化介质参数
    pa1, pa2, pa3 = 0.0, 0.0, 0.0
    for i in range(nspec):
        pa1 += px[i]
        pa2 += px[i] / pz[i]
        pa3 += px[i] * py[i] / pz[i]

    # 计算介质的组合参数
    pk1 = 1.0 - pa1
    pk2 = 1.0 + pa2
    pk4 = pk1 * pk2
    pk5 = pk2**2 - pa3**2
    pa = pk1 * copsis + pk2 * sipsis
    pb = -pk4 * (copsis + 1.0) - pk5 * sipsis
    pc = pk1 * pk5
    pd = abs(pb**2 - 4.0 * pa * pc)

    # 初始化变量
    pmu = 1.0
    gmu = 1.0
    pmus = 0.0

    # 计算折射率
    if pd < 0.0:
        if path[0] < re + 90.0:
            pmu = 1.0
            gmu = 1.0
    else:
        pmus = (-pb - np.sqrt(pd)) / (2.0 * pa)
        if pmus >= 0.0:
            pmu = np.sqrt(pmus)
            if path[0] < re + 90.0:
                pmu = 1.0
                gmu = 1.0

    return (pmu0,
            copsi, copsis, sipsi, sipsis, psi_angl,
            px, py, pz,
            pa1, pa2, pa3, pk1,
            pk2, pk4, pk5,
            pa, pb, pc, pd,
            pmu, gmu, pmus)


def funct(path, dpath, imdfy, nspec, fkc, xlpp):
    """
    Python 实现的 FUNCT 函数，用于计算路径和路径导数。

    参数：
    - path: numpy 数组，表示当前路径 [RR, TH, PH, KX, KY, KZ]
    - dpath: numpy 数组，用于存储路径导数
    - imdfy: 修改标志位，1 表示需要修改路径方向
    - nspec: 等离子体物种数目
    - fkc: 等离子体频率 (kHz)
    - xlpp: 等离子体层边界的径向位置

    返回：
    - dpath: 更新后的路径导数
    - del_val: DEL 的值
    - eps_val: EPS 的值
    - alp: 介质参数
    - ray: 介质参数
    - pis_angl: 波矢与磁场的夹角
    """

    ##基本参数
    rad = np.pi / 180.0  # 角度转弧度的系数
    deg = 1.0 / rad # 弧度转角度的系数
    cvlcty = 300.0  # 光速 [km/s]
    ##基本参数

    #调用函数
    n_s, dnsdrr, dnsdth, dnsdph = dens(path, nspec, xlpp)
    pmu0, copsi, copsis, sipsi, sipsis, psi_angl, px, py, pz, pa1, pa2, pa3, pk1, pk2, pk4, pk5, pa, pb, pc, pd, pmu, gmu, pmus = ref(path, nspec, xlpp, fkc)
    yr, yt, yp,dyrdrr, dytdrr, dypdrr, dyrdth, dytdth, dypdth, dyrdph, dytdph, dypdph, bb, dbbdrr, dbbdth, dbbdph = dipole(path)
    #若 IMDFY==1，则对速度分量(对应 PATH[3], PATH[4], PATH[5])进行缩放处理
    #并计算 DEL, EPS (供记录或后续分析)
    if imdfy == 1:
        path[3] *= (pmu / pmu0)
        path[4] *= (pmu / pmu0)
        path[5] *= (pmu / pmu0)  

        DEL_val = np.arctan2(path[4], path[3])
        EPS_val = np.arctan(path[5] / np.sqrt(path[3]**2 + path[4]**2))

        del_val = DEL_val * deg  # 弧度->角度
        eps_val = EPS_val * deg  # 弧度->角度

    #初始化中间变量
    dmudrr = 0.0
    dmudth = 0.0
    dumdph = 0.0
    dk2df  = 0.0
    da3df  = 0.0

    denom  = (4.0 * pa * pmus + 2.0 * pb) * pmu

    #主循环：对 NSPEC 个物种求和
    for i in range(nspec):
        # (1) 计算 DADX, DBDX, DCDX 等
        dadx = -copsis
        dbdx = -(pk1 / pz[i] - pk2) * (1.0 + copsis)
        dcdx = -pk5
        pk2a3y = 2.0 * (pk2 - pa3 * py[i])

        dadx += (sipsis / pz[i])
        dbdx -= (pk2a3y * sipsis / pz[i])
        dcdx += (pk1 * pk2a3y / pz[i])

        dmudx = -( (dadx*(pmus**2)) + (dbdx*pmus) + dcdx ) / denom

        # (2) 计算 DK2DY, DK5DY, DADY, DBDY, DCDY, DMUDY
        dk2dy = -2.0 * px[i] * py[i] / (pz[i]**2)
        dk5dy =  2.0 * pa3 * px[i] * (pz[i]+2.0) / (pz[i]**2)
        dk5dy += 2.0 * dk2dy * pk2

        dady = dk2dy * sipsis
        dbdy = -pk1*dk2dy*(1.0 + copsis) - dk5dy*sipsis
        dcdy =  pk1*dk5dy

        dmudy = -( (dady*(pmus**2)) + (dbdy*pmus) + dcdy ) / denom

        # (3) 空间坐标梯度：DXDRR, DXDTH, ...
        dxdrr = px[i] * dnsdrr[i]
        dxdth = px[i] * dnsdth[i]
        dxdph = px[i] * dnsdph[i]
        dydrr = py[i] * dbbdrr
        dydth = py[i] * dbbdth
        dydph = py[i] * dbbdph

        dmudrr += (dmudx*dxdrr + dmudy*dydrr)
        dmudth += (dmudx*dxdth + dmudy*dydth)
        dumdph += (dmudx*dxdph + dmudy*dydph)

        # 计算 dk2dfi, da3dfi, 并累加到 dk2df, da3df
        dk2dfi = 2.0 * px[i] / (pz[i]**2)
        da3dfi = (2.0 - pz[i]) * px[i] * py[i] / (pz[i]**2)

        dk2df += dk2dfi / fkc
        da3df += da3dfi / fkc

    # 计算 DMUDPS
    dadps  = pa1 + pa2
    dbdps  = pk4 - pk5
    dmudps = (dadps * pmus + dbdps) * pmus
    dmudps = -2.0 * copsi * dmudps / denom

    #计算 DPSDRR, DPSDTH, DPSDPH (对应 Fortran)
    #注意 PATH[3]->PATH(4), PATH[4]->PATH(5), PATH[5]->PATH(6) 的索引
    dpsdrr = (dyrdrr * path[3] + dytdrr * path[4] + dypdrr * path[5])
    dpsdth = (dyrdth * path[3] + dytdth * path[4] + dypdth * path[5])
    dpsdph = (dyrdph * path[3] + dytdph * path[4] + dypdph * path[5])

    dpsdrr = -dpsdrr / pmu
    dpsdth = -dpsdth / pmu
    dpsdph = -dpsdph / pmu

    # 判断 N_S(1)，更新 DMUDRR, DMUDTH, DMUDPH, DMUDVR, DMUDVT, DMUDVP
    if n_s[0] != 0.0:
        dmudrr = (dmudrr + dmudps * dpsdrr) / pmu
        dmudth = (dmudth + dmudps * dpsdth) / pmu
        dumdph = (dumdph + dmudps * dpsdph) / pmu

        dmudvr = dmudps * (path[3]*copsi / pmu - yr)
        dmudvt = dmudps * (path[4]*copsi / pmu - yt)
        dmudvp = dmudps * (path[5]*copsi / pmu - yp)
    else:
        dmudrr = 0.0
        dmudth = 0.0
        dumdph = 0.0
        dmudvr = 0.0
        dmudvt = 0.0
        dmudvp = 0.0

    #计算 DK1DF, DK4DF, DK5DF, DCDF, DADF, DBDF, DMUDF
    dk1df = 2.0 * pa1 / fkc
    dk4df = pk1 * dk2df + pk2 * dk1df
    dk5df = 2.0 * (pk2 * dk2df - pa3 * da3df)
    dcdf  = pk1 * dk5df + pk5 * dk1df

    dadf = dk1df*copsis + dk2df*sipsis
    dbdf = -dk4df*(1.0 + copsis) - dk5df*sipsis

    dmudf = -((dadf*pmus + dbdf)*pmus + dcdf) / denom
    if n_s[0] == 0.0:
        dmudf = 0.0

    # 更新 GMU
    gmu = pmu + fkc * dmudf

    if path[0] < 6460.0:
        dmudvr = 0.0
        dmudvp = 0.0
        dmudrr = 0.0
        dmudth = 0.0
        dumdph = 0.0

    dpath[0] = path[3] - dmudvr
    dpath[1] = path[4] - dmudvt
    dpath[2] = path[5] - dmudvp

    if pmus != 0.0:
        dpath[0] /= pmus
        if path[0] != 0.0:
            dpath[1] /= (pmus * path[0])
        else:
            dpath[1] = 0.0

        sin_val = np.sin(rad * path[1])
        if abs(sin_val) > 1.0e-12 and path[0] != 0.0:
            dpath[2] /= (pmus * path[0] * sin_val)
        else:
            dpath[2] = 0.0
    else:
        dpath[0] = 0.0
        dpath[1] = 0.0
        dpath[2] = 0.0


    dpath[3] = dmudrr + path[4]*dpath[1] + path[5]*dpath[2]*np.sin(rad*path[1])
    if path[0] != 0.0:
        dpath[4] = (dmudth / path[0]
                    - (path[4]*dpath[0]/path[0])
                    + path[5]*dpath[2]*np.cos(rad*path[1]))
    else:
        dpath[4] = 0.0

    if (path[0] != 0.0) and (abs(sin_val) > 1.0e-12):
        dpath[5] = (dumdph / path[0] / sin_val
                    - path[5]*dpath[0]/path[0]
                    - path[5]*dpath[1]/np.tan(rad*path[1]))
    else:
        dpath[5] = 0.0

    #即将 dθ, dφ 转化为角度
    dpath[1] *= deg
    dpath[2] *= deg

    if pmu != 0.0:
        dpath[6] = 1.0 / pmu / cvlcty
    else:
        dpath[6] = 0.0
    dpath[7] = 1.0 / cvlcty

    #若 n_s(1)==0.0 则 alf=0.0，否则计算 alf
    if n_s[0] == 0.0:
        alf = 0.0

    if pmu != 0.0:
        alf = deg * ( - np.arctan( dmudps * sipsi / pmu ) )
    else:
        alf = 0.0

    ray = psi_angl + alf
    #未输出，记得处理

    return dpath, denom

def integrate_adams_spherical(funct, initial_path, step_size, max_steps, nspec, fkc, xlpp):
    """
    在球坐标(r,theta,phi,kx,ky,kz, extra1, extra2)下 使用自适应 Adams 积分方法，
    同时具有动态调整阶数和步长的功能，并在步长过小时退出积分。
    此处使用相对误差衡量局部误差，当误差超过1e6时，
    会大幅缩小步长并重新计算当前步来降低误差。

    新增异常检测逻辑：
      1. 检测路径中的 NaN 值
      2. 检测半径突变（单步变化超过50%）
      3. 检查折射率有效性（pmu>0且非 NaN）
      4. 检测波矢模长异常
    """
    # 初始化路径与历史记录
    path = np.array(initial_path, dtype=float)
    dpath = np.zeros_like(path)
    trajectory = [path.copy()]
    prev_dpaths = []

    # 绝对误差容差设置
    tol_min = 1e-8
    tol_max = 1e-1

    # 最小允许步长：步长太小则退出积分
    min_step = 1e-6

    # 定义不同阶数的 Adams-Bashforth 系数（预测器：阶数从2到12）
    # 数值为示例近似值
    ab_coeffs = {
        2: np.array([1.5, -0.5]),
        3: np.array([1.8333333333, -1.3333333333, 0.5]),
        4: np.array([2.0833333333, -2.0, 1.3333333333, -0.3333333333]),
        5: np.array([2.2833333333, -2.4833333333, 2.0833333333, -1.0833333333, 0.2833333333]),
        6: np.array([2.45, -2.7833333333, 2.5, -1.65, 0.55, -0.0833333333]),
        7: np.array([2.5928571429, -3.0380952381, 2.65, -1.9, 0.95, -0.2380952381, 0.0285714286]),
        8: np.array([2.720, -3.274, 2.940, -2.220, 1.580, -0.820, 0.200, -0.020]),
        9: np.array([2.837, -3.490, 3.100, -2.580, 1.850, -1.050, 0.450, -0.090, 0.010]),
        10: np.array([2.945, -3.693, 3.270, -2.840, 2.150, -1.400, 0.700, -0.200, 0.030, -0.002]),
        11: np.array([3.045, -3.884, 3.430, -3.030, 2.450, -1.700, 1.050, -0.450, 0.120, -0.015, 0.001]),
        12: np.array([3.138, -4.065, 3.580, -3.170, 2.700, -2.050, 1.500, -0.800, 0.250, -0.035, 0.003, -0.0002]),
    }

    # 定义不同阶数的 Adams-Moulton 系数（校正器：阶数从2到12）
    # 数值为示例近似值
    am_coeffs = {
        2: np.array([0.5, 0.5]),
        3: np.array([0.4166666667, 0.5, -0.0833333333]),
        4: np.array([0.375, 0.7916666667, -0.3125, 0.0625]),
        5: np.array([0.3472222222, 0.7291666667, -0.3958333333, 0.1458333333, -0.0243055556]),
        6: np.array([0.325, 0.681, -0.406, 0.179, -0.040, 0.003]),
        7: np.array([0.307, 0.645, -0.412, 0.222, -0.091, 0.017, -0.001]),
        8: np.array([0.293, 0.615, -0.407, 0.246, -0.114, 0.036, -0.005, 0.0003]),
        9: np.array([0.281, 0.589, -0.395, 0.260, -0.134, 0.048, -0.010, 0.0012, -0.00006]),
        10: np.array([0.270, 0.568, -0.378, 0.271, -0.149, 0.062, -0.015, 0.002, -0.0001, 0.000005]),
        11: np.array([0.261, 0.550, -0.362, 0.279, -0.162, 0.073, -0.020, 0.003, -0.0003, 0.00001, -0.0000006]),
        12: np.array([0.253, 0.536, -0.348, 0.284, -0.173, 0.080, -0.025, 0.004, -0.0005, 0.00002, -0.000001, 0.00000007]),
    }

    # 当前使用的阶数，允许范围 2 到 12
    current_order = 2
    # 初始化初始步长
    current_step = step_size

    for nloop in range(max_steps):
        # 检查步长是否过小，若过小则退出积分
        if current_step < min_step:
            print(f"Step size too small ({current_step:.2e}) at step {nloop}. Exiting integration.")
            break

        # 启动阶段：使用 RK4 方法直到累计足够的历史导数
        if len(prev_dpaths) < current_order:
            # RK4 四阶方法
            k1, demon1 = funct(path, dpath, 1, nspec, fkc, xlpp)
            if np.isnan(k1).any() or demon1 == 0:
                print(f"Error in RK4 stage 1 at step {nloop}")
                break
            k2, demon2 = funct(path + current_step * k1 / 2, dpath, 1, nspec, fkc, xlpp)
            if np.isnan(k2).any() or demon2 == 0:
                print(f"Error in RK4 stage 2 at step {nloop}")
                break
            k3, demon3 = funct(path + current_step * k2 / 2, dpath, 1, nspec, fkc, xlpp)
            if np.isnan(k3).any() or demon3 == 0:
                print(f"Error in RK4 stage 3 at step {nloop}")
                break
            k4, demon4 = funct(path + current_step * k3, dpath, 1, nspec, fkc, xlpp)
            if np.isnan(k4).any() or demon4 == 0:
                print(f"Error in RK4 stage 4 at step {nloop}")
                break

            dpath = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            path += current_step * dpath
            prev_dpaths.append(dpath.copy())

        else:
            # 使用 Adams-Bashforth 预测器（当前阶数）
            ab_coeff = ab_coeffs[current_order]
            recent_dpaths = prev_dpaths[-current_order:]
            ab_pred = sum(c * d for c, d in zip(ab_coeff, recent_dpaths))
            predict_path = path + current_step * ab_pred

            # 计算预测点的导数
            pred_dpath, demon = funct(predict_path, np.zeros_like(dpath), 0, nspec, fkc, xlpp)
            if np.isnan(pred_dpath).any() or demon == 0:
                print(f"Error in prediction at step {nloop}")
                break

            # 使用 Adams-Moulton 校正器（当前阶数）
            am_coeff = am_coeffs[current_order]
            am_corr = am_coeff[0] * pred_dpath
            if current_order > 1:
                am_corr += sum(c * d for c, d in zip(am_coeff[1:], recent_dpaths[-(current_order-1):]))
            
            # 如果am_corr的数值异常，直接退出积分
            if np.max(np.abs(am_corr)) > 1e2 * ab_pred.max():
                # 这里的 1e2 是一个经验值，可以根据实际情况调整
                print(f"am_corr magnitude too high ({np.max(np.abs(am_corr)):.4e}) at step {nloop}. Exiting integration.")
                break

            # 误差估计：使用绝对误差衡量预测值与校正值之间的差异
            error = np.max(np.abs(ab_pred - am_corr))/np.max(np.abs(am_corr))
            # 计算相对误差

            # 如果误差极大（例如超过1e6），则认为当前步计算出现突变，
            # 需要大幅缩小步长并重新计算
            if error > 1e2:
                current_step *= 0.001  # 大幅缩小步长
                print(f"Error extremely high ({error:.4e}) at step {nloop}. "
                      f"Reducing step size drastically to {current_step:.4e} and recalculating step.")
                continue

            # 根据绝对误差调整阶数
            if error > tol_max and current_order < 12:
                current_order = min(current_order + 1, 12)
                print(f"Increasing order to {current_order} at step {nloop}")
            elif error < tol_min and current_order > 2:
                current_order = max(current_order - 1, 2)
                print(f"Decreasing order to {current_order} at step {nloop}")

            # 动态调整步长：若误差过大则缩小步长；若误差非常小则适当增大步长
            if error > tol_max:
                factor = (tol_max / error)**(1.0 / current_order)
                factor = max(factor, 0.5)
                current_step *= factor
                print(f"Reducing step size to {current_step:.4e} at step {nloop} due to high error {error:.4e}")
            elif error < tol_min:
                factor = (tol_min / error)**(1.0 / current_order)
                factor = min(factor, 1.5)
                current_step *= factor
                print(f"Increasing step size to {current_step:.4e} at step {nloop} due to low error {error:.4e}")

            # 更新路径
            path += current_step * am_corr

            # 更新历史导数，记录长度不超过 12
            prev_dpaths.append(pred_dpath.copy())
            if len(prev_dpaths) > 12:
                prev_dpaths.pop(0)

        # 重新计算导数，用于下一步
        dpath, demon = funct(path, dpath, 1, nspec, fkc, xlpp)
        if np.isnan(dpath).any() or demon == 0:
            print(f"Error in derivative calculation at step {nloop}")
            break

        # 保存当前路径
        if np.isnan(path).any():
            print(f"Error in path at step {nloop}")
            break
        trajectory.append(path.copy())

        # 检查积分终止条件：半径限制
        current_r = path[0]
        if current_r < 6370.0 or current_r > 1.0e5:
            print(f"Path out of bounds (radius {current_r} km) at step {nloop}")
            break

        # 检查半径突变（与前一步比较）
        if len(trajectory) >= 2:
            prev_r = trajectory[-2][0]
            delta_r = abs(current_r - prev_r)
            if prev_r != 0 and delta_r > 0.5 * prev_r:
                print(f"Abnormal radius jump ({delta_r:.1f} km) at step {nloop}. Stopping.")
                break

        # 检查 ref 函数返回的折射率有效性（pmu）
        ref_result = ref(path, nspec, xlpp, fkc)
        pmu = ref_result[-3]  # pmu 为 ref 返回的倒数第三个值
        if pmu <= 0 or np.isnan(pmu):
            print(f"Invalid refractive index (pmu={pmu}) at step {nloop}. Stopping.")
            break

        # 检查波矢模长异常
        k_magnitude = np.sqrt(path[3]**2 + path[4]**2 + path[5]**2)
        if k_magnitude > 3e4 or k_magnitude < 1e-3:
            print(f"Abnormal wave vector magnitude ({k_magnitude}) at step {nloop}. Stopping.")
            break

    return np.array(trajectory)


def main():
    # 初始状态 path=[r, theta, phi, k_r, k_th, k_ph, Extra1, Extra2]
    initial_path = [6370.0 + 300, 55.0, 0.0, 0.1, -0.0174, 0.0, 0.0, 0.0]
    step_size = 10.0
    max_steps = 100000
    nspec = 4
    fkc = 10.0
    xlpp = 4.0

    traj = integrate_adams_spherical(funct, initial_path, step_size, max_steps, nspec, fkc, xlpp)
    print("Trajectory shape:", traj.shape)
    
    # 绘图
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal')
    plt.grid(True)

    # 转换到笛卡尔坐标
    RE = 6370.0
    rad = np.pi/180.0
    r_array = traj[:,0]/RE  # 归一化到地球半径
    th_array = traj[:,1]*rad
    x_array = r_array*np.sin(th_array)
    z_array = r_array*np.cos(th_array)

    # 绘制射线轨迹
    plt.plot(x_array, z_array, 'r-', linewidth=1, label='Ray path')
    plt.plot(x_array[0], z_array[0], 'r*', label='Start point')

    # 绘制地球
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    plt.plot(np.cos(theta), np.sin(theta), 'k-')

    # 绘制磁力线(L壳层)
    for L in range(2, 7):
        theta = np.linspace(-np.pi/2, np.pi/2, 1000)
        r = L * np.cos(theta)**2
        valid = r >= 1
        plt.plot(r[valid] * np.cos(theta[valid]), 
                r[valid] * np.sin(theta[valid]), 
                'k:', linewidth=0.5)

    # 绘制等离子体层顶
    L = xlpp
    theta = np.linspace(-np.pi/2, np.pi/2, 1000)
    r = L * np.cos(theta)**2
    valid = r >= 1
    plt.plot(r[valid] * np.cos(theta[valid]), 
            r[valid] * np.sin(theta[valid]), 
            'k--', linewidth=1)

    # 设置图形属性 
    plt.xlim([0, 6])
    plt.ylim([-2.5, 2.5])
    plt.xlabel('[RE]', fontsize=14)
    plt.ylabel('[RE]', fontsize=14)
    plt.title(f'Trajectory at {fkc} kHz with angle = -10°', fontsize=14)
    plt.grid(True)
    plt.legend()

    # 保存图形
    plt.savefig(f'trajectory_{fkc}kHz_angle_-10.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    main()
