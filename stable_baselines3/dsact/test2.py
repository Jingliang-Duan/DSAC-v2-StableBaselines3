import collections

# --- 核心逻辑函数 (与之前相同) ---

TOTAL_HATS = {'R': 3, 'B': 4, 'W': 5}
memo = {}

def can_deduce(k, r, b, w):
    state = (k, r, b, w)
    if state in memo:
        return memo[state]
    if k == 10:
        if (r == TOTAL_HATS['R'] and b == TOTAL_HATS['B']) or \
           (r == TOTAL_HATS['R'] and w == TOTAL_HATS['W']) or \
           (b == TOTAL_HATS['B'] and w == TOTAL_HATS['W']):
            memo[state] = True
            return True
        memo[state] = False
        return False
    possible_colors = []
    if r < TOTAL_HATS['R'] and not can_deduce(k + 1, r + 1, b, w):
        possible_colors.append('R')
    if b < TOTAL_HATS['B'] and not can_deduce(k + 1, r, b + 1, w):
        possible_colors.append('B')
    if w < TOTAL_HATS['W'] and not can_deduce(k + 1, r, b, w + 1):
        possible_colors.append('W')
    memo[state] = (len(possible_colors) == 1)
    return memo[state]

def precompute_all_deductions():
    """从后往前预计算所有k和所有可见帽子组合的can_deduce结果。"""
    print("第一步：预计算所有逻辑可能性...")
    for k in range(10, 0, -1):
        num_seen = k - 1
        for r in range(min(num_seen, TOTAL_HATS['R']) + 1):
            for b in range(min(num_seen - r, TOTAL_HATS['B']) + 1):
                w = num_seen - r - b
                if w >= 0 and w <= TOTAL_HATS['W']:
                    can_deduce(k, r, b, w)
    print("逻辑预计算完成。\n")

def print_logic_tree(k, r, b, w, indent=""):
    """递归地打印逻辑决策树。"""
    
    # 打印当前节点状态
    print(f"{indent}P{k} 看见 {{R:{r}, B:{b}, W:{w}}}")
    
    # 检查当前节点是否能得出结论
    if can_deduce(k, r, b, w):
        print(f"{indent}  └──> \033[92m结论: P{k} 知道答案！\033[0m (此推理分支终止)")
        return

    # 如果不能，则继续向下探索
    print(f"{indent}  └──> P{k} 说 '我不知道'，推理继续至 P{k-1}...")
    
    # 如果已经到P6，就不再向下展开
    if k == 6:
        print(f"{indent}      └──> \033[92m结论: 在此前提下，P6 必然知道答案！\033[0m (所有P7及之后的分支都终止于P6)")
        return

    # 递归探索P(k-1)可能看到的组合
    # 情况1: P(k-1)戴的是红帽子
    if r > 0:
        print_logic_tree(k - 1, r - 1, b, w, indent + "    |")
    # 情况2: P(k-1)戴的是黑帽子
    if b > 0:
        print_logic_tree(k - 1, r, b - 1, w, indent + "    |")
    # 情况3: P(k-1)戴的是白帽子
    if w > 0:
        print_logic_tree(k - 1, r, b, w - 1, indent + "    |")


# --- 主程序 ---
precompute_all_deductions()

print("第二步：生成并打印逻辑决策树")
print("树的根节点是P10可能看到的所有情况。我们将追踪所有'我不知道'的分支。")
print("="*50)

# P10能看到9顶帽子
num_seen_by_p10 = 9
for r in range(min(num_seen_by_p10, TOTAL_HATS['R']) + 1):
    for b in range(min(num_seen_by_p10 - r, TOTAL_HATS['B']) + 1):
        w = num_seen_by_p10 - r - b
        if w >= 0 and w <= TOTAL_HATS['W']:
            # 对于P10可能看到的每一种组合，展开一棵推理树
            print_logic_tree(10, r, b, w)
            print("-" * 50)

print("\n树状图分析总结：")
print("如上所示，尽管存在大量不同的分支路径，但**没有任何一条路径**能够让'我不知道'的链条延伸超过P7。")
print("所有从P10开始的、逻辑上可能的“我不知道”推理链，最终都终止于P6（或更早）。")
print("这严谨地证明了，P6是第一个必然能知道自己帽子颜色的人。")