import functools

# 帽子总数
TOTAL_HATS = {'R': 3, 'B': 4, 'W': 5}

# 记忆化，存储 (k, r, b, w) 状态下，第k个人是否能知道答案
memo = {}

def can_deduce(k, r, b, w):
    """
    判断第k个人(1-10)在看到r, b, w组合时能否推断出自己的帽子颜色。
    k: 人的编号 (10是最后面的人, 1是第一个)
    r, b, w: 第k个人看到的在他前面(1 to k-1)的帽子数量
    """
    state = (k, r, b, w)
    if state in memo:
        return memo[state]

    # P10的逻辑 (基础情况，没有来自后面的信息)
    if k == 10:
        if (r == TOTAL_HATS['R'] and b == TOTAL_HATS['B']) or \
           (r == TOTAL_HATS['R'] and w == TOTAL_HATS['W']) or \
           (b == TOTAL_HATS['B'] and w == TOTAL_HATS['W']):
            memo[state] = True
            return True
        memo[state] = False
        return False

    # P1到P9的逻辑
    possible_colors = []
    
    # 假设1: Pk的帽子是红色
    if r < TOTAL_HATS['R']:
        if not can_deduce(k + 1, r + 1, b, w):
            possible_colors.append('R')

    # 假设2: Pk的帽子是黑色
    if b < TOTAL_HATS['B']:
        if not can_deduce(k + 1, r, b + 1, w):
            possible_colors.append('B')

    # 假设3: Pk的帽子是白色
    if w < TOTAL_HATS['W']:
        if not can_deduce(k + 1, r, b, w + 1):
            possible_colors.append('W')

    result = len(possible_colors) == 1
    memo[state] = result
    return result

def find_max_unknown_chain(k, r, b, w):
    """
    寻找从后往前最长的“不知道”链条的长度。
    返回链条长度和该链条对应的P1的帽子颜色。
    """
    if can_deduce(k, r, b, w):
        return 0, None

    if k == 1:
        hat_color = ''
        if r > 0: hat_color = 'R'
        elif b > 0: hat_color = 'B'
        else: hat_color = 'W'
        return 1, hat_color

    max_sub_chain = -1
    p1_hat_for_max_sub_chain = None
    
    if r > 0:
        l, h = find_max_unknown_chain(k - 1, r - 1, b, w)
        if l > max_sub_chain:
            max_sub_chain = l
            p1_hat_for_max_sub_chain = h
    if b > 0:
        l, h = find_max_unknown_chain(k - 1, r, b - 1, w)
        if l > max_sub_chain:
            max_sub_chain = l
            p1_hat_for_max_sub_chain = h
    if w > 0:
        l, h = find_max_unknown_chain(k - 1, r, b, w - 1)
        if l > max_sub_chain:
            max_sub_chain = l
            p1_hat_for_max_sub_chain = h

    if max_sub_chain == -1:
        return 1, None

    return 1 + max_sub_chain, p1_hat_for_max_sub_chain

# --- 主程序 ---
print("正在分析所有可能的帽子组合...")
max_overall_chain_len = -1
final_p1_hat_identity = None
worst_case_config = None

for r_total in range(TOTAL_HATS['R'] + 1):
    for b_total in range(TOTAL_HATS['B'] + 1):
        for w_total in range(TOTAL_HATS['W'] + 1):
            if r_total + b_total + w_total == 10:
                memo.clear()
                current_max_len_for_config = -1
                current_p1_hat_for_config = None

                if r_total > 0:
                    r_seen, b_seen, w_seen = r_total - 1, b_total, w_total
                    l, h = find_max_unknown_chain(10, r_seen, b_seen, w_seen)
                    if l > current_max_len_for_config:
                        current_max_len_for_config = l
                        current_p1_hat_for_config = h
                
                if b_total > 0:
                    r_seen, b_seen, w_seen = r_total, b_total - 1, w_total
                    l, h = find_max_unknown_chain(10, r_seen, b_seen, w_seen)
                    if l > current_max_len_for_config:
                        current_max_len_for_config = l
                        current_p1_hat_for_config = h

                if w_total > 0:
                    r_seen, b_seen, w_seen = r_total, b_total, w_total - 1
                    l, h = find_max_unknown_chain(10, r_seen, b_seen, w_seen)
                    if l > current_max_len_for_config:
                        current_max_len_for_config = l
                        current_p1_hat_for_config = h

                if current_max_len_for_config > max_overall_chain_len:
                    max_overall_chain_len = current_max_len_for_config
                    final_p1_hat_identity = current_p1_hat_for_config
                    worst_case_config = (r_total, b_total, w_total)

person_who_knows_idx = 10 - max_overall_chain_len
num_of_questions_asked = max_overall_chain_len + 1
print(f"在最坏情况下，第{person_who_knows_idx}个人知道答案。")
print(f"帽子组合为: {worst_case_config}, P1的帽子颜色为: {final_p1_hat_identity}")

print("分析完成！")