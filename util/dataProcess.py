### Generate processable data from raw data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
#   1.分割数据集
def split_data(filename):
    # 读取 Excel 文件
    file_path = filename  # 替换为你的文件路径
    df = pd.read_excel(file_path)

    # 按 30% 和 70% 分割数据
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # 保存分割后的数据为 Excel 文件
    train_df.to_excel('train_data2.xlsx', index=False)
    test_df.to_excel('test_data2.xlsx', index=False)

    print(f"{file_path}" + "已分割完成")

def concat_excel(filename1, filename2,filename3):
    df1 = pd.read_excel(filename1)
    df2 = pd.read_excel(filename2)
    df3 = pd.read_excel(filename3)

    # 合并三个 DataFrame，按行合并（即将数据叠加）
    merged_df = pd.concat([df1, df2, df3], ignore_index=True)

    # 将合并后的数据保存到新的 Excel 文件
    merged_df.to_excel('train.xlsx', index=False)

    print("三个 Excel 文件已成功")

#   2.找到每个长度所对应的海滩剖面(x,y)

def match_data(filename1, filename2,output):
    # 读取两个 Excel 文件
    file1_path = filename1  # 替换为实际路径
    file2_path = filename2  # 替换为实际路径

    # 读取 Excel 文件中的数据
    df1 = pd.read_excel(file1_path)
    df2 = pd.read_excel(file2_path)

    # 提取 df2 中的第四列和第五列
    df2_columns = df2.iloc[:, [3, 4]]  # 第4列和第5列

    # 逐行比较 df1 和 df2 的第四列和第五列
    matching_rows = []
    for i, row in df1.iterrows():
        # 获取 df1 当前行的第四列和第五列
        df1_values = row.iloc[3:5].values  # 第4列和第5列的值
        # 检查 df1 当前行的第四列和第五列是否在 df2 中匹配
        if any((df1_values == df2_values).all() for df2_values in df2_columns.values):
            matching_rows.append(row)

    # 将匹配的行保存到新的 Excel 文件
    matched_df = pd.DataFrame(matching_rows)
    matched_df.to_excel(output, index=False)

    print(f"匹配的行已保存到 {output}")

#   3.标准化
###  对角度求sin，对每一列进行标准化
def standardize_data(filename, output):
    file_path = filename
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 对第4列和第14列取 sin
    df.iloc[:, 3] = np.sin(df.iloc[:, 3])  # 第4列取sin
    df.iloc[:, 13] = np.sin(df.iloc[:, 13])  # 第14列取sin

    # 对第4列到第23列进行标准化处理
    # 选择第4列到第23列
    columns_to_standardize = df.iloc[:, 3:23]

    # 使用 StandardScaler 进行标准化
    scaler = StandardScaler()
    df.iloc[:, 3:23] = scaler.fit_transform(columns_to_standardize)

    # 将处理后的结果保存到新的 Excel 文件
    df.to_excel(output, index=False)

    print(f"处理后的数据已保存到{output}")
