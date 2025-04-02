from typing import List, Dict, Set, Tuple, Union
import json
import os
import zhipuai
from datetime import datetime, timezone
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# from data.process import dataset

class DrugKGProcessor:
    def __init__(self, api_key: str, dataset: str):
        """
        初始化药物知识图谱处理器
        Args:
            api_key: 智谱AI的API key
            dataset: 数据集名称
        """
        self.dataset = dataset
        # 设置文件路径
        self.input_folder = f"/home/zyh0023/code/Z_RecDrug/data/drugs_info/{dataset}/"
        self.block_folder = f"/home/zyh0023/code/Z_RecDrug/create_KG/drugs_block/{dataset}/"
        self.output_folder = f"/home/zyh0023/code/Z_RecDrug/create_KG/Drugs_KG/{dataset}/"
        # 确保输出目录存在
        os.makedirs(self.block_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # 初始化知识图谱提取器
        self.kg_extractor = KGExtractor(api_key=api_key, max_concurrent=200)

        # 创建必要的目录
        for folder in [self.input_folder, self.block_folder, self.output_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                # print(f"Created directory: {folder}")

    def extract_blocks(self, drug_json_file: str) -> str:
        """
        将单个药物的完整JSON文件拆分成块
        Args:
            drug_json_file: 完整的药物JSON文件路径
        Returns:
            blocks_folder: 生成的blocks文件夹路径
        """
        try:
            # 读取JSON文件
            with open(drug_json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取药物ATC编码（从文件名中提取）
            atc_code = os.path.splitext(os.path.basename(drug_json_file))[0]

            # 创建该药物的blocks文件夹
            blocks_folder = os.path.join(self.block_folder, atc_code, 'blocks')
            if not os.path.exists(blocks_folder):
                os.makedirs(blocks_folder)

            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 为每个字段创建单独的JSON文件
            for field, value in data.items():
                file_path = os.path.join(blocks_folder, f"{field}.json")
                try:
                    field_data = {
                        "field": field,
                        "content": value
                    }
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(field_data, f, indent=2, ensure_ascii=False)
                    # print(f"Created block file: {file_path}")

                except Exception as e:
                    print(f"Error saving {field}: {str(e)}")

            return blocks_folder

        except Exception as e:
            print(f"Error processing {drug_json_file}: {str(e)}")
            return ""

    async def process_all_drugs(self):
        """处理所有药物文件"""
        # 获取输入文件夹中的所有JSON文件
        json_files = [f for f in os.listdir(self.input_folder) if f.endswith('.json')]

        print(f"Found {len(json_files)} drug files to process")

        for json_file in tqdm(json_files, desc="Processing JSON files", unit="file"):
            drug_json_path = os.path.join(self.input_folder, json_file)
            # print(f"\nProcessing drug file: {json_file}")

            # 1. 提取blocks
            blocks_folder = self.extract_blocks(drug_json_path)
            if not blocks_folder:
                print(f"Failed to extract blocks for {json_file}")
                continue

            # 2. 为该药物生成知识图谱
            atc_code = os.path.splitext(json_file)[0]
            try:
                print(f"\nGenerating knowledge graph for {atc_code}")
                status = await self.kg_extractor.process_file_async(
                    folder_path=os.path.dirname(blocks_folder),
                    output_path=self.output_folder
                )

                print(f"Completed processing {atc_code}")
                # print(f"Status: {'Success' if status['processed'] else 'Failed'}")
                if status.get('error'):
                    print(f"Error: {status['error']}")

            except Exception as e:
                print(f"Error generating knowledge graph for {atc_code}: {str(e)}")

class KGExtractor:
    def __init__(self, api_key: str, max_concurrent: int = 200):
        """
        初始化知识图谱提取器
        Args:
            api_key: 智谱AI的API key
            ax_concurrent: 最大并发数
        """
        # 更新模型名称为新版API支持的名称
        self.client = zhipuai.ZhipuAI(api_key=api_key)  # 创建客户端实例并存储
        self.model = "glm-4-flash"  # 或使用 "chatglm_std", "chatglm_lite"
        self.retries = 3  # 重试次数
        self.retry_delay = 2  # 重试延迟(秒)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _call_glm_async(self, prompt: str) -> str:
        """异步调用ChatGLM API"""
        async with self.semaphore:  # 使用信号量控制并发
            for attempt in range(self.retries):
                try:
                    # 将同步API调用包装在to_thread中
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.model,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        top_p=0.7,
                    )
                    # print('response:', response)

                    if response and hasattr(response, 'choices') and response.choices:
                        return response.choices[0].message.content
                    else:
                        print("API Error: Unexpected response format")

                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.retries - 1:
                        await asyncio.sleep(self.retry_delay)
                    continue

            return ""

    async def extract_entities_async(self, text: str) -> Set[str]:
        """Extract entities from text"""
        prompt = f"""
        Please extract key entities from the following drug-related text. Focus on:
        1. Drug names and synonyms
        2. Chemical substances
        3. Mechanisms of action
        4. Therapeutic uses
        5. Biological processes
        6. Related diseases
        7. Drug classifications
        8. Drug targets
        9. Side effects
        10. Metabolic pathways

        Only return a JSON array of entities. Each entity should be precise and medically relevant.

        Text:
        {text}

        Return in this format:
        ["entity1", "entity2", "entity3"]

        Requirements:
        - Extract only meaningful medical/pharmaceutical terms
        - Keep original technical terms as they appear in the text
        - Include both generic and brand names if present
        - Include specific molecular targets and pathways
        - Exclude general words and non-medical terms
        """

        try:
            response = await self._call_glm_async(prompt)
            if not response.strip().startswith('['):
                response = response[response.find('['):response.rfind(']')+1]
            entities = json.loads(response)
            return set(entities)
        except json.JSONDecodeError:
            print("Failed to parse entities response")
            return set()

    async def extract_relations_async(self, text: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities"""
        prompt = f"""
        Analyze the following drug-related text and identify relationships between the entities.
        Known entities: {list(entities)}

        Extract relationship triples in the format: (subject, relationship, object)
        Focus on these relationship types:
        1. Drug-Target: "inhibits", "activates", "binds to", "modulates"
        2. Drug-Disease: "treats", "indicated for", "prevents"
        3. Drug-Mechanism: "acts via", "works through", "metabolized by"
        4. Drug-Effect: "causes", "results in", "leads to"
        5. Drug-Classification: "belongs to", "is a type of", "classified as"
        6. Drug-Pathway: "involved in", "affects pathway", "regulates"

        Text:
        {text}

        Return ONLY a JSON array of triples:
        [
            ["subject1", "relationship1", "object1"],
            ["subject2", "relationship2", "object2"]
        ]

        Requirements:
        - Use ONLY entities from the provided entity list
        - Relationships should be specific and biochemically accurate
        - Keep relationships concise and clear
        - Ensure all subjects and objects are from the known entities list
        - Focus on pharmacologically relevant relationships
        """
        # 5. Drug-Interaction: "interacts with", "enhances", "reduces"

        try:
            response = await self._call_glm_async(prompt)
            if not response.strip().startswith('['):
                response = response[response.find('['):response.rfind(']')+1]
            relations = json.loads(response)
            valid_relations = [
                rel for rel in relations
                if rel[0] in entities and rel[2] in entities
            ]
            return valid_relations
        except json.JSONDecodeError:
            print("Failed to parse relations response")
            return []

    async def extract_drug_info_async(self, drug_data: dict) -> str:
        """异步版本的药物信息提取"""
        info_parts = []

        # 添加基本信息
        info_parts.append(f"Drug: {drug_data.get('name', '')}")
        info_parts.append(f"DrugBank ID: {drug_data.get('drugbank_id', '')}")

        # 添加关键字段的信息
        key_fields = [
            'description', 'indication', 'pharmacodynamics',
            'mechanism_of_action', 'toxicity', 'metabolism',
            'absorption'
        ]

        for field in key_fields:
            value = drug_data.get(field, '')
            if value and value != "N/A":
                info_parts.append(f"{field.title()}: {value}")

        # 添加同义词
        synonyms = drug_data.get('synonyms', [])
        if synonyms:
            info_parts.append(f"Synonyms: {', '.join(synonyms)}")

        # 添加ATC编码
        atc_codes = drug_data.get('atc_codes', [])
        if atc_codes:
            info_parts.append(f"ATC Codes: {', '.join(atc_codes)}")

        return "\n".join(info_parts)

    async def process_item_async(self, drug_id: str, drug_data: Union[str, Dict]) -> Dict:
        """异步处理单个药物条目"""
        try:
            # 构建文本内容
            if isinstance(drug_data, str):
                text = f"ID: {drug_id}\nContent: {drug_data}"
            elif isinstance(drug_data, dict):
                text = await self.extract_drug_info_async(drug_data)
            elif isinstance(drug_data, list):
                text = f"ID: {drug_id}\nContent: {', '.join(map(str, drug_data))}"
            else:
                print(f"Skipping {drug_id}: unexpected data type")
                return None

            if not text.strip():
                print(f"Skipping {drug_id}: empty content")
                return None

            # print(f"\nProcessing ID {drug_id}...")
            # print("Extracting entities...")
            entities = await self.extract_entities_async(text)

            if not entities:
                print(f"No entities found for {drug_id}")
                return None

            # print("Extracting relations...")
            relations = await self.extract_relations_async(text, entities)

            print(f"Successfully processed {drug_id}")
            # print(f"Found {len(entities)} entities and {len(relations)} relations")

            return {
                "drug_id": drug_id,
                "entities": list(entities),
                "relations": relations
            }

        except Exception as e:
            print(f"Error processing {drug_id}: {str(e)}")
            return None

    async def process_document_async(self, data: Dict[str, Union[str, Dict, List]]) -> Dict:
        """异步处理所有药物数据"""
        all_entities = set()
        all_relations = []
        processed_items = 0
        failed_items = 0

        try:
            # print(f"Starting async processing with {len(data)} items")

            # 创建所有任务并使用信号量控制并发
            tasks = []
            for drug_id, drug_data in data.items():
                task = self.process_item_async(drug_id, drug_data)
                tasks.append(task)

            # 异步执行所有任务
            results = await asyncio.gather(*tasks)

            # 处理结果
            for result in results:
                if result:
                    all_entities.update(result["entities"])
                    all_relations.extend(result["relations"])
                    processed_items += 1
                else:
                    failed_items += 1

            # 构建知识图谱
            edges = set(rel[1] for rel in all_relations)

            return {
                "entities": list(all_entities),
                "relations": all_relations,
                "edges": list(edges),
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "total_items": len(data),
                    "processed_items": processed_items,
                    "failed_items": failed_items,
                    "entity_count": len(all_entities),
                    "relation_count": len(all_relations),
                    "edge_type_count": len(edges)
                }
            }

        except Exception as e:
            print(f"Error in async processing: {str(e)}")
            return {
                "entities": [],
                "relations": [],
                "edges": [],
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "error": str(e),
                    "total_items": len(data),
                    "processed_items": processed_items,
                    "failed_items": failed_items
                }
            }

    async def validate_entities(self, all_entities: set, file_path: str) -> set:
        """
        验证实体是否存在于当前处理的JSON文件中
        Args:
            self: KGExtractor实例
            all_entities: 提取的实体集合
            file_path: 当前处理的JSON文件路径
        Returns:
            验证后的实体集合
        """
        valid_entities = set()
        all_content = ""

        try:
            # print(f"\nValidating {len(all_entities)} entities in file: {file_path}")

            # 读取当前JSON文件
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 获取content内容
                    if isinstance(data, dict) and "content" in data:
                        content = data["content"]
                        if isinstance(content, list):
                            # 如果是列表，将所有项转换为字符串
                            all_content = " ".join(
                                json.dumps(item, ensure_ascii=False) if isinstance(item, dict)
                                else str(item)
                                for item in content
                            )
                        else:
                            all_content = str(content)

            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                return all_entities

            # 对内容进行预处理
            all_content = all_content.lower()

            # 验证每个实体
            total_entities = len(all_entities)
            validated_count = 0
            removed_count = 0

            # print("\nStarting entity validation...")

            for entity in all_entities:
                # 获取实体的基本形式（去除括号内容）并转换为小写
                entity_lower = entity.lower()
                base_entity = entity_lower.split('(')[0].strip()

                # 生成可能的变体（处理单复数）
                variations = {base_entity}  # 基本形式
                if base_entity.endswith('s'):
                    variations.add(base_entity[:-1])  # 单数形式
                else:
                    variations.add(base_entity + 's')  # 复数形式

                # 检查任何一个变体是否在原文中出现
                found = any(var in all_content for var in variations)

                if found:
                    valid_entities.add(entity)  # 保留原始形式（包括括号和大小写）
                    validated_count += 1
                else:
                    removed_count += 1
                    # print(f"Removed entity: {entity} (not found in current file)")

            # 打印验证结果
            # print(f"\nEntity Validation Results for {os.path.basename(file_path)}:")
            # print(f"Total entities: {total_entities}")
            # print(f"Valid entities: {validated_count}")
            # print(f"Removed entities: {removed_count}")

        except Exception as e:
            print(f"Error during entity validation: {str(e)}")
            return all_entities

        return valid_entities


    async def process_file_async(self, folder_path: str, output_path: str):
        """异步处理文件夹下的所有JSON文件"""
        drug_name = ""  # 初始化药物名称
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        status = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "processed": False,
            "error": None,
            "start_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            # 获取所有JSON文件
            # print("folder_path:", folder_path)
            folder_path = os.path.join(folder_path, "blocks")
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            # 排除不需要处理的文件
            skip_files = ['drugbank_id.json', 'group.json', 'references.json']
            json_files = [f for f in json_files if f not in skip_files]
            status["total_files"] = len(json_files)

            all_entities = set()
            all_relations = []

            # 首先处理 name.json 获取药物名称
            name_file = "name.json"
            if name_file in json_files:
                name_path = os.path.join(folder_path, name_file)
                try:
                    with open(name_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "content" in data:
                            drug_name = data["content"]
                except Exception as e:
                    print(f"Error processing name.json: {str(e)}")

            for filename in json_files:
                file_path = os.path.join(folder_path, filename)
                field_name = filename.replace('.json', '')  # 获取字段名（文件名不含扩展名）

                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "content" in data:
                            content = data["content"]
                        else:
                            continue

                    # 根据字段类型处理,结构化字段特殊处理
                    special_fields = ['atc_codes', 'enzymes', 'synonyms', 'targets', 'drug_interactions']

                    if field_name in special_fields:
                        # 对特殊字段，将字段名作为实体
                        field_entity = field_name
                        all_entities.add(field_entity)

                        if field_name == 'synonyms':
                            # 直接将 content 列表中的每个项添加为实体
                            entities = set(content)  # 将列表转换为集合去重
                            all_entities.update(entities)
                            # 可以添加一个关系表示这些都是同义词
                            for entity in entities:
                                all_relations.append([drug_name, "has synonym", entity])


                        elif field_name == 'drug_interactions':
                            if isinstance(content, list):
                                for interaction in content:
                                    # 从每个交互对象中提取药物名称
                                    if isinstance(interaction, dict):
                                        interacting_drug = interaction.get('name', '')
                                        description = interaction.get('description', '')

                                        # 将交互药物添加到实体集合
                                        if interacting_drug:
                                            all_entities.add(interacting_drug)
                                            # 添加药物交互关系
                                            all_relations.append([drug_name, "interacts with", interacting_drug])
                                            # print("[drug_name interacts_with interacting_drug]:", [drug_name, "interacts_with", interacting_drug])

                                        # # 从描述中提取额外的实体
                                        # if description:
                                        #     entities = await self.extract_entities_async(description)
                                        #     validated_entities = await self.validate_entities(entities, file_path)
                                        #     all_entities.update(validated_entities)

                                            # # 可以添加更详细的关系，例如描述交互的性质
                                            # for entity in validated_entities:
                                            #     if entity != interacting_drug and entity != drug_name:
                                            #         relations = await self.extract_relations_async(description, validated_entities)
                                            #         all_relations.append(relations)
                                            #         print('[find_relations]:', relations)
                                # print(f"Processed {len(content)} drug interactions")

                    else:
                        # 其他文件的处理保持不变
                        text = str(content)
                        new_entities = await self.extract_entities_async(text)
                        validated_new_entities = await self.validate_entities(new_entities, file_path)
                        all_entities.update(validated_new_entities)
                        relations = await self.extract_relations_async(text, validated_new_entities)
                        all_relations.extend(relations)

                    status["processed_files"] += 1
                    # print(f"Processed {filename} \n")

                except Exception as e:
                    error_msg = f"Error processing {filename}: {str(e)}"
                    print(error_msg)
                    status["error"] = error_msg
                    status["failed_files"] += 1

            # 生成知识图谱
            kg = {
                "drug_name": drug_name,  # 添加药物名称
                "entities": list(all_entities),
                "relations": all_relations,
                "edges": list(set(rel[1] for rel in all_relations)),
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "total_files": status["total_files"],
                    "processed_files": status["processed_files"],
                    "failed_files": status["failed_files"],
                    "entity_count": len(all_entities),
                    "relation_count": len(all_relations)
                }
            }

            # 获取ATC编码（从folder_path中提取）
            atc_code = os.path.basename(os.path.dirname(folder_path))

            # 保存知识图谱
            output_file = os.path.join(output_path, f"{atc_code}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(kg, f, ensure_ascii=False, indent=2)
            # # 保存知识图谱
            # output_file = os.path.join(output_path, f"{drug_name}_kg.json")
            # with open(output_file, 'w', encoding='utf-8') as f:
            #     json.dump(kg, f, ensure_ascii=False, indent=2)

            status["processed"] = status["processed_files"] > 0
            print(f"\nSuccessfully generated knowledge graph of {drug_name}")

        except Exception as e:
            status["error"] = f"Error in process_file_async: {str(e)}"
            status["processed"] = False

        finally:
            status["end_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            # 保存状态
            status_file = os.path.join(output_path, "processing_status.json")
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)

        return status


async def main():
    API_KEY = "25c9d1a5b45d43e9874eb695a3d9ce09.xgGIAxyzmUSlhVvn"  # 替换为你的API key
    DATASETS = ['mimic-iii', 'mimic-iv']

    # 添加数据集处理的总进度条
    with tqdm(DATASETS, desc="Processing datasets", unit="dataset") as dataset_pbar:
        for DATASET in dataset_pbar:
            dataset_pbar.set_description(f"Processing dataset: {DATASET}")

            # 创建提取器实例
            processor = DrugKGProcessor(api_key=API_KEY, dataset=DATASET)

            # 处理所有药物
            await processor.process_all_drugs()
            dataset_pbar.write(f"Completed processing dataset: {DATASET}")


if __name__ == "__main__":
    asyncio.run(main())




