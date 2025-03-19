import dill
import os
import glob
import re
import shutil
import xml.etree.ElementTree as ET

def filter_drugbank_data(med_voc, drugbank_dir, output_dir):
    """
    从 DrugBank 数据集中筛选符合条件的药物：
    - 仅保留 `approved` 状态的药物
    - 仅匹配 `med_voc` 中的 ATC-4 编码
    - 只保留指定字段
    - 以 `med_voc.word2idx[ATC-4]` 作为文件名保存

    参数：
    voc_path (str): med_voc 词汇表的路径
    drugbank_dir (str): DrugBank XML 文件存放的目录
    output_dir (str): 处理后数据的存放目录
    """

    # 计算药物总数
    num_medications = len(med_voc.word2idx)
    print(f"The number of medications in {dataset} med_voc:", num_medications)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 DrugBank 目录中的所有 XML 文件
    drugbank_files = glob.glob(os.path.join(drugbank_dir, "*.txt"))
    matched_files = 0

    for file in drugbank_files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()

            # 仅保留 'approved' 药物
            drug_groups = root.findall(".//groups/group")
            if not any(group.text == "approved" for group in drug_groups):
                continue

            # 获取 ATC 代码并筛选 ATC-4 级别
            atc_codes = root.findall(".//atc-codes/atc-code")
            drug_atc4 = set()

            for atc in atc_codes:
                atc_code = atc.attrib.get('code', '')  # 提取 ATC 代码
                if len(atc_code) >= 7:  # 至少是 ATC-4 级别
                    atc4_code = atc_code[:4]  # 取 ATC-4 编码
                    drug_atc4.add(atc4_code)

            # 判断当前药物是否符合 med_voc 中的 ATC-4
            matched_atc4 = drug_atc4.intersection(med_voc.word2idx.keys())

            if matched_atc4:
                # 仅保留以下字段
                selected_fields = [
                    "drugbank-id", "name", "description", "indication", "pharmacodynamics",
                    "mechanism-of-action", "toxicity", "metabolism", "absorption", "classification",
                    "synonyms", "atc-codes", "drug-interactions", "sequences", "enzymes", "pathways",
                    "reactions"
                ]

                selected_data = ET.Element("drug")

                for field in selected_fields:
                    element = root.find(f".//{field}")
                    if element is not None:
                        selected_data.append(element)

                # 处理 <targets>，仅保留 <name> 和 <actions>
                targets_element = root.find(".//targets")
                if targets_element is not None:
                    new_targets = ET.Element("targets")
                    for target in targets_element.findall(".//target"):
                        target_name = target.find(".//name")
                        actions = target.find(".//actions")
                        new_target = ET.Element("target")
                        if target_name is not None:
                            new_target.append(target_name)
                        if actions is not None:
                            new_target.append(actions)
                        if len(new_target) > 0:
                            new_targets.append(new_target)
                    selected_data.append(new_targets)


                # 获取 ATC-4 作为文件名
                for atc4 in matched_atc4:
                    file_name = f"{med_voc.word2idx[atc4]}.txt"  # 用序列号作为文件名
                    file_path = os.path.join(output_dir, file_name)

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(ET.tostring(selected_data, encoding="utf-8").decode("utf-8"))

                matched_files += 1

        except ET.ParseError:
            print(f"解析 XML 失败，跳过文件：{file}")  # 处理格式异常的文件

    print(f"筛选完成，共匹配到 {matched_files} 个 DrugBank 文件")
    print(f"筛选后的文件存放在 {output_dir} 目录下")


def calculate_match_percentage(med_voc, drugbank_dir):
    """
    计算 med_voc 中的 ATC-4 代码在 DrugBank 数据集中匹配的比例。

    参数：
    voc_path (str): med_voc 词汇表的路径
    drugbank_dir (str): DrugBank XML 文件存放的目录

    返回：
    float: 匹配比例（匹配数 / 总数）
    """

    # 计算 med_voc 中 ATC-4 编码的总数
    total_medications = len(med_voc.word2idx)
    matched_medications = set()

    # 遍历 DrugBank XML
    drugbank_files = glob.glob(os.path.join(drugbank_dir, "*.txt"))

    for file in drugbank_files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()

            # 仅保留 'approved' 药物
            drug_groups = root.findall(".//groups/group")
            if not any(group.text == "approved" for group in drug_groups):
                continue

            # 获取 ATC-4 级别的代码
            atc_codes = root.findall(".//atc-codes/atc-code")
            for atc in atc_codes:
                atc_code = atc.attrib.get('code', '')
                if len(atc_code) >= 4:
                    atc4_code = atc_code[:4]
                    if atc4_code in med_voc.word2idx:
                        matched_medications.add(atc4_code)

        except ET.ParseError:
            print(f"解析 XML 失败，跳过文件：{file}")

    match_ratio = len(matched_medications) / total_medications if total_medications > 0 else 0
    print(f"匹配的药物数: {len(matched_medications)} / {total_medications}")
    return match_ratio



# 加载 voc_final.pkl 中的 med_voc 数据
# 开始
for dataset in ['mimic-iii', 'mimic-iv']:
    print("-" * 10, "processing dataset: ", dataset, "-" * 10)
    voc_path = f'/home/zyh0023/code/Z_RecDrug/data/output/{dataset}/voc_final.pkl'
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']  # 获取 med_voc 数据

    # 定义 DrugBank XML 路径和输出路径
    drugbank_dir = "/home/zyh0023/code/Z_RecDrug/data/drugbank_output/"
    output_dir = f"/home/zyh0023/code/Z_RecDrug/data/drugs_info/{dataset}/"

    # 筛选 DrugBank 数据
    filter_drugbank_data(med_voc, drugbank_dir, output_dir)

    # 计算匹配度
    match_percentage = calculate_match_percentage(med_voc, drugbank_dir)
    print(f"匹配度: {match_percentage:.2%}")