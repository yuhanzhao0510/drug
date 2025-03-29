import json
import os
import dill
from collections import defaultdict


def filter_and_save_drugbank_data(med_voc, drugbank_json_path, output_dir):
    """
    从 DrugBank JSON 数据集中筛选药物并保存，确保准确的计数
    """
    # 计算词汇表中的药物总数
    num_medications = len(med_voc.word2idx)
    print(f"The number of medications in med_voc:", num_medications)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取 DrugBank JSON 文件
    with open(drugbank_json_path, 'r') as f:
        drugbank_data = json.load(f)

    # 用于存储匹配结果的字典
    drug_to_save = {}  # key: atc5_code, value: drug_info
    atc4_to_drugs = defaultdict(set)  # key: atc4, value: set of drug_ids

    # 为每个药物选择一个合适的ATC-5编码
    for drug_id, drug_info in drugbank_data.items():
        if drug_info.get('group') != 'approved':
            continue

        atc_codes = sorted(drug_info.get('atc_codes', []))  # 排序以确保一致性
        selected_atc5 = None

        # 找出第一个匹配的ATC-5编码
        for atc_code in atc_codes:
            if len(atc_code) >= 7:  # 确保长度足够
                atc4_code = atc_code[:4]
                if atc4_code in med_voc.word2idx:
                    selected_atc5 = atc_code[:7]
                    atc4_to_drugs[atc4_code].add(drug_id)
                    drug_to_save[selected_atc5] = drug_info
                    break  # 找到第一个匹配就停止

    # 保存匹配到的药物信息
    saved_count = 0
    for atc5_code, drug_info in drug_to_save.items():
        file_name = f"{atc5_code}.json"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(drug_info, f, ensure_ascii=False, indent=2)
        saved_count += 1

    # 创建汇总文件
    summary_file = os.path.join(output_dir, "atc_codes_summary.json")
    summary_data = {
        "total_matched_drugs": saved_count,
        "total_atc4_codes": len(atc4_to_drugs),
        "atc4_statistics": {
            atc4: len(drugs)
            for atc4, drugs in atc4_to_drugs.items()
        },
        "file_count": saved_count
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"筛选完成，共保存了 {saved_count} 个唯一药物信息文件")
    print(f"涉及 {len(atc4_to_drugs)} 个不同的ATC-4编码")
    return saved_count, len(atc4_to_drugs)


def calculate_match_statistics(med_voc, drugbank_json_path):
    """
    计算匹配统计信息
    """
    total_medications = len(med_voc.word2idx)
    matched_drugs = set()
    matched_atc4 = set()

    with open(drugbank_json_path, 'r') as f:
        drugbank_data = json.load(f)

    # 使用字典来跟踪每个药物的ATC5编码
    drug_atc5_mapping = {}

    for drug_id, drug_info in drugbank_data.items():
        if drug_info.get('group') != 'approved':
            continue

        atc_codes = sorted(drug_info.get('atc_codes', []))  # 排序以确保一致性
        for atc_code in atc_codes:
            if len(atc_code) >= 7:
                atc4_code = atc_code[:4]
                if atc4_code in med_voc.word2idx:
                    matched_drugs.add(drug_id)
                    matched_atc4.add(atc4_code)
                    drug_atc5_mapping[drug_id] = atc_code[:7]
                    break  # 只取第一个匹配的ATC编码

    print(f"匹配的唯一药物数量: {len(matched_drugs)}")
    print(f"匹配的ATC-4编码数: {len(matched_atc4)} / {total_medications}")
    return len(matched_drugs), len(matched_atc4)


# 主程序
if __name__ == "__main__":
    for dataset in ['mimic-iii', 'mimic-iv']:
        print("-" * 10, f"Processing dataset: {dataset}", "-" * 10)
        voc_path = f'/home/zyh0023/code/Z_RecDrug/data/output/{dataset}/voc_final.pkl'
        voc = dill.load(open(voc_path, 'rb'))
        med_voc = voc['med_voc']

        drugbank_json_path = "./drugbank_all_drugs.json"
        output_dir = f"/home/zyh0023/code/Z_RecDrug/data/drugs_info/{dataset}/"

        # 筛选并保存药物数据
        saved_count, atc4_count = filter_and_save_drugbank_data(med_voc, drugbank_json_path, output_dir)

        # 验证实际保存的文件数量
        actual_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and f != 'atc_codes_summary.json']
        print(f"\n文件验证:")
        print(f"计数的文件数量: {saved_count}")
        print(f"实际的文件数量: {len(actual_files)}")

        # 计算详细统计信息
        unique_drugs, matched_atc4_count = calculate_match_statistics(med_voc, drugbank_json_path)
        print(f"\n{dataset} 详细统计信息:")
        print(f"匹配的唯一药物数: {unique_drugs}")
        print(f"匹配的ATC-4编码数: {matched_atc4_count}")