import re
import xml.etree.ElementTree as ET
import os
import json
import concurrent.futures
from tqdm import tqdm
from io import StringIO


# 定义 XML 解析的辅助函数
def get_text(element):
    """ 安全获取 XML 元素的文本内容 """
    return element.text.strip() if element is not None and element.text else "N/A"


def parse_drug_element(drug):
    """
    解析单个药物元素，返回药物数据字典
    """
    drug_id = get_text(drug.find(".//drugbank-id[@primary='true']"))
    drug_name = get_text(drug.find(".//name"))
    description = get_text(drug.find(".//description"))
    indication = get_text(drug.find(".//indication"))
    pharmacodynamics = get_text(drug.find(".//pharmacodynamics"))
    mechanism_of_action = get_text(drug.find(".//mechanism-of-action"))
    toxicity = get_text(drug.find(".//toxicity"))
    metabolism = get_text(drug.find(".//metabolism"))
    absorption = get_text(drug.find(".//absorption"))
    classification = get_text(drug.find(".//classification/description"))

    # 获取 <synonyms>
    synonyms = [get_text(syn) for syn in drug.findall(".//synonyms/synonym")]

    # 获取 <atc-codes>
    atc_codes = [atc.attrib.get('code', "N/A") for atc in drug.findall(".//atc-codes/atc-code")]

    # 获取 <drug-interactions>
    interactions = []
    for interaction in drug.findall(".//drug-interactions/drug-interaction"):
        interact_name = get_text(interaction.find(".//name"))
        interact_desc = get_text(interaction.find(".//description"))
        interactions.append({
            "name": interact_name,
            "description": interact_desc
        })

    # 获取 <sequences>
    sequences = [get_text(seq) for seq in drug.findall(".//sequences/sequence")]

    # 获取 <enzymes>
    enzymes = [get_text(enzyme.find(".//name")) for enzyme in drug.findall(".//enzymes/enzyme")]

    # 获取 <pathways>
    pathways = [get_text(pathway.find(".//name")) for pathway in drug.findall(".//pathways/pathway")]

    # 获取 <reactions>
    reactions = [get_text(reaction.find(".//name")) for reaction in drug.findall(".//reactions/reaction")]

    # 获取 <targets>
    targets = []
    for target in drug.findall(".//targets/target"):
        target_name = get_text(target.find(".//name"))
        actions = [get_text(action) for action in target.findall(".//actions/action")]
        targets.append({
            "name": target_name,
            "actions": actions if actions else []
        })

    # 只保存参考文献链接信息，不下载
    references = []
    for link in drug.findall(".//links/link"):
        ref_id = get_text(link.find(".//ref-id"))
        title = get_text(link.find(".//title"))
        url = get_text(link.find(".//url"))

        if url != "N/A":
            references.append({
                "ref_id": ref_id,
                "title": title,
                "url": url
            })

    # 组织JSON数据
    drug_info = {
        "drugbank_id": drug_id,
        "name": drug_name,
        "description": description,
        "indication": indication,
        "pharmacodynamics": pharmacodynamics,
        "mechanism_of_action": mechanism_of_action,
        "toxicity": toxicity,
        "metabolism": metabolism,
        "absorption": absorption,
        "classification": classification,
        "synonyms": synonyms,
        "atc_codes": atc_codes,
        "drug_interactions": interactions,
        "sequences": sequences,
        "enzymes": enzymes,
        "pathways": pathways,
        "reactions": reactions,
        "targets": targets,
        "references": references
    }

    return drug_id, drug_info


def process_drugbank_data(xml_input, output_file, max_workers=4):
    """
    处理 DrugBank 数据，使用多线程将所有药物保存到一个JSON文件中

    参数：
    - xml_input (str): DrugBank XML 文件的路径 (full_database.xml)
    - output_file (str): 输出JSON文件路径
    - max_workers (int): 并行工作线程数
    """
    print(f"开始解析 DrugBank XML 文件: {xml_input}")
    tree = ET.parse(xml_input)
    root = tree.getroot()

    # 移除 XML 命名空间前缀
    print("处理XML命名空间...")
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]

    # 解析 XML
    drugs = root.findall("drug")
    print(f"找到 {len(drugs)} 个药物条目")

    # 多线程处理药物数据
    drug_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务到线程池
        future_to_drug = {executor.submit(parse_drug_element, drug): drug for drug in drugs}

        # 处理完成的任务并显示进度条
        with tqdm(total=len(drugs), desc="处理药物数据") as pbar:
            for future in concurrent.futures.as_completed(future_to_drug):
                try:
                    drug_id, drug_info = future.result()
                    drug_data[drug_id] = drug_info
                except Exception as exc:
                    print(f"处理药物时发生错误: {exc}")
                finally:
                    pbar.update(1)

    # 保存为单个JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(drug_data, f, ensure_ascii=False, indent=2)

    print(f"共处理了 {len(drug_data)} 个药物，数据已保存到 {output_file}")


if __name__ == "__main__":
    xml_input = "./full_database.xml"  # 输入 XML 文件路径
    output_directory = "./data"  # 输出目录
    output_file = os.path.join(output_directory, "drugbank_all_drugs.json")  # 单一输出文件

    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 确定线程数（默认为CPU核心数）
    import multiprocessing

    max_workers = max(1, multiprocessing.cpu_count() - 1)

    process_drugbank_data(xml_input, output_file, max_workers)
    print("所有处理完成!")